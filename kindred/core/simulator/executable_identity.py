"""Immutable identities for admitted and target-bound executable science."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)

__all__ = [
    "BaseExecutableScienceIdentity",
    "TargetExecutableScienceIdentity",
    "compose_target_executable_identity",
    "fingerprint_executable_payload",
]


def _required_fingerprint(value: object, field: str, prefix: str) -> str:
    fingerprint = str(value or "").strip()
    if not fingerprint.startswith(prefix) or len(fingerprint) <= len(prefix):
        raise ValueError(f"{field} must be a nonblank {prefix!r} fingerprint.")
    return fingerprint


def fingerprint_executable_payload(payload: object, *, prefix: str) -> str:
    """Hash one already-canonical executable payload."""

    if type(prefix) is not str or not prefix or prefix != prefix.strip():
        raise ValueError("Executable fingerprint prefix must be nonblank text.")
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{prefix}{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class BaseExecutableScienceIdentity:
    """Parser-minted identity of admitted science before target binding."""

    fingerprint: str
    analysis_identity: str
    standard_concentration_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint",
            _required_fingerprint(
                self.fingerprint,
                "base executable fingerprint",
                "science-base:",
            ),
        )
        analysis = str(self.analysis_identity or "").strip()
        if not analysis:
            raise ValueError("base executable analysis identity must be nonblank.")
        object.__setattr__(self, "analysis_identity", analysis)
        standard = str(self.standard_concentration_identity or "").strip()
        if not standard.startswith("C0:"):
            raise ValueError(
                "standard concentration identity must be a C0 fingerprint."
            )
        object.__setattr__(self, "standard_concentration_identity", standard)


@dataclass(frozen=True, slots=True)
class TargetExecutableScienceIdentity:
    """Target-owner composition of base science and one bound intervention."""

    base: BaseExecutableScienceIdentity
    intervention_declarative_fingerprint: str
    intervention_executable_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.base, BaseExecutableScienceIdentity):
            raise TypeError("target executable identity requires typed base science.")
        for field in (
            "intervention_declarative_fingerprint",
            "intervention_executable_fingerprint",
        ):
            value = getattr(self, field)
            if type(value) is not str or value != value.strip():
                raise ValueError(f"{field} must be an edge-normalized string.")
        object.__setattr__(
            self,
            "fingerprint",
            _required_fingerprint(
                self.fingerprint,
                "target executable fingerprint",
                "science-target:",
            ),
        )


def compose_target_executable_identity(
    base: BaseExecutableScienceIdentity,
    intervention_execution: InterventionExecutionContract,
) -> TargetExecutableScienceIdentity:
    """Compose target science without inspecting or reconstructing a mechanism."""

    if not isinstance(base, BaseExecutableScienceIdentity):
        raise TypeError("base must be BaseExecutableScienceIdentity.")
    if not isinstance(intervention_execution, InterventionExecutionContract):
        raise TypeError(
            "target executable identity requires InterventionExecutionContract."
        )
    declarative = intervention_execution.declarative_fingerprint
    executable = intervention_execution.executable_fingerprint
    fingerprint = fingerprint_executable_payload(
        {
            "base": base.fingerprint,
            "intervention_declarative": declarative,
            "intervention_executable": executable,
        },
        prefix="science-target:",
    )
    return TargetExecutableScienceIdentity(
        base=base,
        intervention_declarative_fingerprint=declarative,
        intervention_executable_fingerprint=executable,
        fingerprint=fingerprint,
    )
