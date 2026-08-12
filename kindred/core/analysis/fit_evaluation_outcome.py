"""Typed outcomes for scientific Fit evaluations.

An optimizer needs a finite vector for every objective call, including a
penalty vector for an admitted candidate-domain rejection.  That vector is
not scientific evidence.  This small envelope keeps the optimizer-facing
numeric representation separate from the scientific outcome consumed by
Profile, Bootstrap, Hierarchical, and terminal reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
    SimulationCancelled,
)

FitEvaluationStatus = Literal[
    "converged",
    "nonconverged",
    "candidate_domain_rejected",
    "cancelled",
    "fatal",
]

_VALID_STATUSES = frozenset(
    {
        "converged",
        "nonconverged",
        "candidate_domain_rejected",
        "cancelled",
        "fatal",
    }
)


@dataclass(frozen=True, slots=True)
class FitEvaluationOutcome:
    """One typed Fit outcome with finite scientific evidence when available.

    ``residuals`` is deliberately *not* the optimizer's penalty vector.  It
    is present only when a scientific residual evaluation completed.  An
    optimizer may still return a finite penalty array to SciPy, but the
    candidate-domain outcome remains attached to the objective and therefore
    cannot be mistaken for likelihood evidence.
    """

    status: FitEvaluationStatus
    residuals: np.ndarray | None = None
    transport_residuals: np.ndarray | None = None
    cause: BaseException | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported Fit evaluation status: {status!r}")
        cause = self.cause
        if status == "candidate_domain_rejected":
            if cause is None or not isinstance(cause, FitSimulationError):
                raise TypeError(
                    "Candidate-domain outcomes require their typed FitSimulationError cause."
                )
        elif status == "cancelled":
            if cause is not None and not isinstance(
                cause,
                (FittingCancelled, SimulationCancelled),
            ):
                raise TypeError(
                    "Cancelled Fit outcomes require a cancellation cause."
                )
        elif status == "fatal" and cause is None:
            raise ValueError("Fatal Fit outcomes require their cause.")

        evidence = None
        if self.residuals is not None:
            evidence = np.asarray(self.residuals, dtype=float).reshape(-1)
            if evidence.size == 0 or not np.all(np.isfinite(evidence)):
                raise ValueError(
                    "Fit scientific residual evidence must be finite and nonempty."
                )
            evidence = evidence.copy()
            evidence.setflags(write=False)
        transport = None
        if self.transport_residuals is not None:
            transport = np.asarray(self.transport_residuals, dtype=float).reshape(-1)
            if transport.size == 0 or not np.all(np.isfinite(transport)):
                raise ValueError(
                    "Fit optimizer transport residuals must be finite and nonempty."
                )
            transport = transport.copy()
            transport.setflags(write=False)
        if status in {"candidate_domain_rejected", "cancelled", "fatal"} and evidence is not None:
            raise ValueError(
                f"Fit outcome {status!r} cannot carry scientific residual evidence."
            )
        if status in {"converged", "nonconverged"} and evidence is None:
            raise ValueError(
                f"Fit outcome {status!r} requires scientific residual evidence."
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "residuals", evidence)
        object.__setattr__(self, "transport_residuals", transport)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance or {})))

    def cause_payload(self) -> Mapping[str, object] | None:
        """Return a stable structured cause without flattening its message."""

        if self.cause is None:
            return None
        details = getattr(self.cause, "details", None)
        return MappingProxyType(
            {
                "type": type(self.cause).__name__,
                "code": str(getattr(self.cause, "code", "") or ""),
                "message": str(getattr(self.cause, "message", None) or self.cause),
                "details": dict(details) if isinstance(details, Mapping) else {},
            }
        )

    @classmethod
    def converged(
        cls,
        residuals: object,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        return cls("converged", np.asarray(residuals, dtype=float), provenance=provenance or {})

    @classmethod
    def nonconverged(
        cls,
        residuals: object,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        return cls(
            "nonconverged",
            np.asarray(residuals, dtype=float),
            provenance=provenance or {},
        )

    @classmethod
    def candidate_rejected(
        cls,
        cause: FitSimulationError,
        *,
        transport_residuals: object | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        return cls(
            "candidate_domain_rejected",
            transport_residuals=(
                None
                if transport_residuals is None
                else np.asarray(transport_residuals, dtype=float)
            ),
            cause=cause,
            provenance=provenance or {},
        )

    @classmethod
    def cancelled(
        cls,
        cause: BaseException | None = None,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        return cls("cancelled", cause=cause, provenance=provenance or {})

    @classmethod
    def fatal(
        cls,
        cause: BaseException,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        return cls("fatal", cause=cause, provenance=provenance or {})

    @classmethod
    def from_exception(
        cls,
        cause: BaseException,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        """Classify a typed execution exception without message inspection."""

        if isinstance(cause, (FittingCancelled, SimulationCancelled)):
            return cls.cancelled(cause, provenance=provenance)
        if isinstance(cause, FitSimulationError):
            details = getattr(cause, "details", {})
            if (
                isinstance(details, Mapping)
                and details.get("candidate_domain") is True
                and details.get("fatal") is False
            ):
                return cls.candidate_rejected(cause, provenance=provenance)
        return cls.fatal(cause, provenance=provenance)


__all__ = ["FitEvaluationOutcome", "FitEvaluationStatus"]
