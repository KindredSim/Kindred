"""Call-local scientific-unavailability facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from kindred.core.simulation_failure import FactualCause, freeze_factual_value

@dataclass(frozen=True, slots=True)
class ScientificUnavailable:
    """Immutable producer-owned explanation for missing candidate evidence.

    This is a semantic result, not an exception classifier. Producers create
    it at the exact source-owned predicate and consumers inspect this immutable
    factual explanation directly.
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
        object.__setattr__(self, "details", freeze_factual_value(dict(self.details or {})))
        if self.factual_cause is not None and not isinstance(
            self.factual_cause,
            FactualCause,
        ):
            raise TypeError("Scientific-unavailability factual cause must be typed.")



__all__ = [
    "ScientificUnavailable",
]
