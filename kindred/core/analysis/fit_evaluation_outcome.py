"""Typed outcomes for scientific Fit evaluations.

An optimizer needs a finite vector for every objective call, including a
penalty vector for an admitted scientific-unavailability outcome.  That vector is
not scientific evidence.  This small envelope keeps the optimizer-facing
numeric representation separate from the scientific outcome consumed by
Profile, Bootstrap, and terminal reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from kindred.core.exceptions import FittingCancelled, SimulationCancelled
from kindred.core.simulation_failure import FactualCause, factual_cause_from_exception

FitEvaluationStatus = Literal[
    "converged",
    "nonconverged",
    "scientific_unavailable",
    "cancelled",
    "fatal",
]

SeriesEvaluationStatus = Literal[
    "available",
    "scientific_unavailable",
    "cancelled",
    "fatal",
]

_VALID_STATUSES = frozenset(
    {
        "converged",
        "nonconverged",
        "scientific_unavailable",
        "cancelled",
        "fatal",
    }
)


@dataclass(frozen=True, slots=True)
class ScientificUnavailable:
    """Immutable producer-owned explanation for missing candidate evidence.

    This is a semantic result, not an exception classifier.  Producers create
    it at the exact source-owned predicate (for example, an unsuccessful SciPy
    solve or non-finite candidate arithmetic), and consumers only inspect the
    immutable status carried by :class:`FitEvaluationOutcome`.
    """

    producer: str
    stage: str
    details: Mapping[str, object] = field(default_factory=dict)
    factual_cause: FactualCause | None = None

    def __post_init__(self) -> None:
        producer = str(self.producer or "").strip()
        stage = str(self.stage or "").strip()
        if not producer or not stage:
            raise ValueError("Scientific-unavailability producer and stage are required.")
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "details", _freeze_value(dict(self.details or {})))
        if self.factual_cause is not None and not isinstance(
            self.factual_cause,
            FactualCause,
        ):
            raise TypeError("Scientific-unavailability factual cause must be typed.")


@dataclass(frozen=True, slots=True)
class FittingSeriesOutcome:
    """Per-dataset transport emitted by a serial fitting evaluator."""

    status: SeriesEvaluationStatus
    result: object | None = None
    unavailability: ScientificUnavailable | None = None
    cause: BaseException | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in {"available", "scientific_unavailable", "cancelled", "fatal"}:
            raise ValueError(f"Unsupported fitting series status: {status!r}")
        if status == "available":
            if self.result is None:
                raise ValueError("Available fitting series outcomes require a result.")
            if self.unavailability is not None or self.cause is not None:
                raise ValueError("Available fitting series outcomes cannot carry failure state.")
        elif status == "scientific_unavailable":
            if self.unavailability is None or self.result is not None or self.cause is not None:
                raise ValueError("Scientific-unavailability series outcomes require only their result.")
        elif status == "cancelled":
            if self.result is not None or self.unavailability is not None:
                raise ValueError("Cancelled fitting series outcomes cannot carry a result.")
            if self.cause is not None and not isinstance(
                self.cause,
                (FittingCancelled, SimulationCancelled),
            ):
                raise TypeError("Cancelled fitting series outcomes require a cancellation cause.")
        else:
            if self.cause is None or self.result is not None or self.unavailability is not None:
                raise ValueError("Fatal fitting series outcomes require only their cause.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "provenance", _freeze_value(dict(self.provenance or {})))

    @classmethod
    def available(cls, result: object, *, provenance: Mapping[str, object] | None = None) -> "FittingSeriesOutcome":
        return cls("available", result=result, provenance=provenance or {})

    @classmethod
    def scientific_unavailable(
        cls,
        unavailability: ScientificUnavailable,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> "FittingSeriesOutcome":
        return cls(
            "scientific_unavailable",
            unavailability=unavailability,
            provenance=provenance or {},
        )

    @classmethod
    def cancelled(cls, cause: BaseException | None = None, *, provenance: Mapping[str, object] | None = None) -> "FittingSeriesOutcome":
        return cls("cancelled", cause=cause, provenance=provenance or {})

    @classmethod
    def fatal(cls, cause: BaseException, *, provenance: Mapping[str, object] | None = None) -> "FittingSeriesOutcome":
        return cls("fatal", cause=cause, provenance=provenance or {})


def _freeze_value(value: object) -> object:
    """Deep-freeze request provenance while retaining mapping semantics."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_value(item) for item in sorted(value, key=repr))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class FitEvaluationOutcome:
    """One typed Fit outcome with finite scientific evidence when available.

    ``residuals`` is deliberately *not* the optimizer's penalty vector.  It
    is present only when a scientific residual evaluation completed.  An
    optimizer may still return a finite penalty array to SciPy, but the
    scientific-unavailability outcome remains attached to the objective and therefore
    cannot be mistaken for likelihood evidence.
    """

    status: FitEvaluationStatus
    residuals: np.ndarray | None = None
    transport_residuals: np.ndarray | None = None
    cause: BaseException | None = None
    unavailability: ScientificUnavailable | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported Fit evaluation status: {status!r}")
        cause = self.cause
        if status == "scientific_unavailable":
            if self.unavailability is None:
                raise TypeError(
                    "Scientific-unavailability outcomes require their producer-owned result."
                )
            if cause is not None:
                raise TypeError(
                    "Scientific-unavailability outcomes cannot carry an exception cause."
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
        if status in {"scientific_unavailable", "cancelled", "fatal"} and evidence is not None:
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
        if self.unavailability is not None and not isinstance(
            self.unavailability, ScientificUnavailable
        ):
            raise TypeError("unavailability must be ScientificUnavailable or None.")
        object.__setattr__(self, "provenance", _freeze_value(dict(self.provenance or {})))
        object.__setattr__(self, "unavailability", self.unavailability)

    def to_factual_cause(self) -> FactualCause | None:
        """Project the execution cause into the canonical typed cause tree."""

        if self.unavailability is not None:
            if self.unavailability.factual_cause is not None:
                return self.unavailability.factual_cause
            return FactualCause(
                domain="fit",
                origin=self.unavailability.producer,
                code=None,
                type="ScientificUnavailable",
                message=(
                    "Scientific result unavailable during "
                    f"{self.unavailability.stage}."
                ),
                details=self.unavailability.details,
            )
        if self.cause is None:
            return None
        return factual_cause_from_exception(
            self.cause,
            domain="fit",
            origin="fit_evaluation",
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
    def scientific_unavailable(
        cls,
        unavailability: ScientificUnavailable,
        *,
        transport_residuals: object | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> "FitEvaluationOutcome":
        return cls(
            "scientific_unavailable",
            transport_residuals=(
                None
                if transport_residuals is None
                else np.asarray(transport_residuals, dtype=float)
            ),
            unavailability=unavailability,
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
        return cls.fatal(cause, provenance=provenance)


__all__ = [
    "FitEvaluationOutcome",
    "FitEvaluationStatus",
    "FittingSeriesOutcome",
    "SeriesEvaluationStatus",
    "ScientificUnavailable",
]
