"""Call-local scientific-unavailability facts."""

from __future__ import annotations

from dataclasses import dataclass

from kindred.core.simulation_failure import FactualCause

@dataclass(frozen=True, slots=True)
class ScientificUnavailable:
    """Immutable producer-owned explanation for missing candidate evidence.

    This is a semantic result, not an exception classifier. Producers create
    it at the exact source-owned predicate and consumers inspect this immutable
    factual explanation directly.
    """

    producer: str
    stage: str
    factual_cause: FactualCause



__all__ = [
    "ScientificUnavailable",
]
