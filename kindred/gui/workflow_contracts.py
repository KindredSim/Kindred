"""Immutable GUI workflow contracts shared by owners and presenters."""

from __future__ import annotations

from dataclasses import dataclass
import math

from kindred.core.mechanism_source import (
    MechanismAuthoringSource,
    source_matches_analysis,
)
from kindred.core.project_document import ProjectDocument
from kindred.core.project_session import ProjectContentToken
from kindred.core.simulator.dsl_parse import DslAnalysis, PreparedDslRequestSet
from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport
from kindred.core.units import UnitsModel

__all__ = [
    "DetailedBalanceCorrectionCommitResult",
    "DetailedBalanceCorrectionOutcome",
    "DetailedBalanceCorrectionProposal",
    "DetailedBalanceCorrectionRequest",
]


@dataclass(frozen=True, slots=True)
class DetailedBalanceCorrectionProposal:
    """One exact-token, mutation-free detailed-balance source proposal."""

    tab_id: str
    source: MechanismAuthoringSource
    expected_token: ProjectContentToken
    proposed_document: ProjectDocument
    prepared_dsl: PreparedDslRequestSet

    def __post_init__(self) -> None:
        tab_id = str(self.tab_id or "").strip()
        if not tab_id:
            raise ValueError("tab_id must be nonblank.")
        if not isinstance(self.source, MechanismAuthoringSource):
            raise TypeError("source must be MechanismAuthoringSource.")
        if not isinstance(self.expected_token, ProjectContentToken):
            raise TypeError("expected_token must be ProjectContentToken.")
        if not isinstance(self.proposed_document, ProjectDocument):
            raise TypeError("proposed_document must be ProjectDocument.")
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
        proposed_tab = self.proposed_document.tab(tab_id)
        if proposed_tab.mechanism_source != self.source:
            raise ValueError(
                "proposal source must be the proposed document's accepted source."
            )
        if proposed_tab.prepared_dsl is not self.prepared_dsl:
            raise ValueError(
                "proposal prepared DSL must be the proposed document's retained DSL."
            )
        if (
            not source_matches_analysis(self.source, self.prepared_dsl.analysis)
        ):
            raise ValueError(
                "proposal prepared DSL must match the accepted source text."
            )
        object.__setattr__(self, "tab_id", tab_id)


@dataclass(frozen=True, slots=True)
class DetailedBalanceCorrectionCommitResult:
    """One canonical-commit result with explicit post-boundary truth."""

    proposal: DetailedBalanceCorrectionProposal
    committed: bool
    content_token: ProjectContentToken | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, DetailedBalanceCorrectionProposal):
            raise TypeError("proposal must be DetailedBalanceCorrectionProposal.")
        if type(self.committed) is not bool:
            raise TypeError("committed must be a bool.")
        if self.content_token is not None and not isinstance(
            self.content_token,
            ProjectContentToken,
        ):
            raise TypeError("content_token must be ProjectContentToken or None.")
        error = self.error
        if error is not None and not isinstance(error, str):
            raise TypeError("error must be str or None.")
        if error is not None and not error.strip():
            raise ValueError("error must be nonblank when provided.")
        if not self.committed and self.content_token is not None:
            raise ValueError("an uncommitted result cannot carry a content token.")


@dataclass(frozen=True, slots=True)
class DetailedBalanceCorrectionRequest:
    """One owner-issued, generation-bound detailed-balance correction request."""

    origin: str
    tab_id: str
    source: str
    issue: str
    accepted_analysis: DslAnalysis
    temperature_K: float = 298.15
    report: WegscheiderCyclicityReport | None = None
    source_energy_unit: str = "kJ/mol"

    def __post_init__(self) -> None:
        for field_name in ("origin", "tab_id", "issue"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be nonblank.")
        if str(self.origin).strip().casefold() not in {"run", "fit", "preview"}:
            raise ValueError("origin must be Run, Fit, or Preview.")
        if not str(self.source or "").strip():
            raise ValueError("source must be nonblank.")
        if not isinstance(self.accepted_analysis, DslAnalysis):
            raise TypeError("accepted_analysis must be DslAnalysis.")
        if not isinstance(self.report, WegscheiderCyclicityReport):
            raise TypeError(
                "Detailed-balance correction requires the accepted cyclicity report."
            )
        if not self.report.unresolved_cycles:
            raise ValueError("Detailed-balance report has no unresolved cycles.")
        temperature = float(self.temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_K must be positive and finite.")
        source_energy_unit = str(self.source_energy_unit or "").strip()
        if not source_energy_unit:
            raise ValueError("source_energy_unit must be nonblank.")
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "source_energy_unit", source_energy_unit)
        if not source_matches_analysis(
            MechanismAuthoringSource.from_parts(reactions_text=self.source),
            self.accepted_analysis,
        ):
            raise ValueError(
                "accepted_analysis must correspond to the exact correction source."
            )

    @property
    def accepted_parse_units(self) -> UnitsModel:
        """Return the parse context that produced the accepted cycle report."""

        return UnitsModel(
            energy_unit=self.source_energy_unit,
            temperature_K=self.temperature_K,
        )


@dataclass(frozen=True, slots=True)
class DetailedBalanceCorrectionOutcome:
    """One immutable owner-fenced result from the correction workflow."""

    origin: str
    tab_id: str
    source: str
    issue: str
    corrected_source: str | None
    cancelled: bool

    def __post_init__(self) -> None:
        for field_name in ("origin", "tab_id", "source", "issue"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be nonblank.")
        if str(self.origin).strip().casefold() not in {"run", "fit", "preview"}:
            raise ValueError("origin must be Run, Fit, or Preview.")
        if type(self.cancelled) is not bool:
            raise TypeError("cancelled must be a bool.")
        corrected = self.corrected_source
        if corrected is not None and not str(corrected).strip():
            raise ValueError("corrected_source must be nonblank when provided.")
        if self.cancelled == (corrected is not None):
            raise ValueError(
                "Detailed-balance outcome must be either cancelled or corrected."
            )
