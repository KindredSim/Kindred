from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from kindred.core.mechanism_source import (
    MechanismAuthoringSource,
    source_matches_analysis,
)
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateContract,
    equilibrium_rate_contract,
)
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.symbolic.calculator import (
    SymbolicCalculatorError,
    evaluate_symbolic_query,
    symbolic_calculator_available,
)


logger = logging.getLogger(__name__)


def _require_executable_fingerprint(value: object) -> str:
    """Require a stable executable identity on every retained calculator item."""

    if type(value) is not str or not value.strip():
        raise ValueError(
            "Symbolic calculator executable fingerprint must be nonblank."
        )
    return value


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorHistoryEntry:
    query: str
    result_text: str
    context_text: str
    base_executable_fingerprint: str
    is_error: bool = False

    def __post_init__(self) -> None:
        _require_executable_fingerprint(self.base_executable_fingerprint)


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorSnapshot:
    source: str
    mechanism: object
    base_executable_fingerprint: str
    equilibrium_summary: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_executable_fingerprint(self.base_executable_fingerprint)


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorViewState:
    available: bool
    reason: str = ""
    latest_query: str = ""
    latest_result_text: str = ""
    latest_is_error: bool = False
    can_copy_result: bool = False
    can_copy_context: bool = False
    equilibrium_summary: tuple[str, ...] = ()


class SymbolicCalculatorOwner:
    def __init__(
        self,
        *,
        canonical_prepared_dsl: Callable[[], PreparedDslRequestSet | None],
        canonical_source: Callable[[], MechanismAuthoringSource],
    ) -> None:
        if not callable(canonical_prepared_dsl):
            raise TypeError("canonical_prepared_dsl must be callable.")
        if not callable(canonical_source):
            raise TypeError("canonical_source must be callable.")
        self._canonical_prepared_dsl = canonical_prepared_dsl
        self._canonical_source = canonical_source
        self._snapshot: SymbolicCalculatorSnapshot | None = None
        self._history: list[SymbolicCalculatorHistoryEntry] = []
        self._available = False
        self._reason = "Canonical mechanism unavailable."

    def bind_canonical_source_resolvers(
        self,
        *,
        canonical_prepared_dsl: Callable[[], PreparedDslRequestSet | None],
        canonical_source: Callable[[], MechanismAuthoringSource],
    ) -> None:
        """Rebind canonical symbolic inputs to the published document owner."""

        self._canonical_prepared_dsl = canonical_prepared_dsl
        self._canonical_source = canonical_source

    def refresh(self) -> SymbolicCalculatorViewState:
        self._snapshot, self._available, self._reason = self._build_snapshot()
        if not self._available and self._reason == "Symbolic calculator failed unexpectedly.":
            self._history.clear()
        return self.view_state()

    def invalidate_current_snapshot(
        self,
        reason: str = "Canonical mechanism has unapplied Reactions.",
    ) -> SymbolicCalculatorViewState:
        """Hide only the current result while retaining immutable history.

        Reactions drafts are a temporary authoring state, not a replacement of
        the calculator owner's history.  Clearing the current snapshot makes
        display and copy unavailable until the accepted DSL is rebuilt, while
        preserving entries that may still match that accepted executable
        fingerprint after the draft is reverted or accepted.
        """

        self._snapshot = None
        self._available = False
        self._reason = str(reason).strip() or "Symbolic calculator unavailable."
        return self.view_state()

    def evaluate(self, query: str) -> SymbolicCalculatorViewState:
        query_s = str(query or "").strip()
        if not query_s:
            return self.view_state()
        self.refresh()
        if not self._available or self._snapshot is None:
            return self.view_state()
        try:
            result = evaluate_symbolic_query(
                self._snapshot.mechanism,
                query_s,
                mechanism_source=self._snapshot.source,
            )
        except SymbolicCalculatorError as exc:
            message = str(exc)
            entry = SymbolicCalculatorHistoryEntry(
                query=query_s,
                result_text=message,
                context_text="\n".join(
                    (
                        f"Query: {query_s}",
                        f"Error: {message}",
                        "Mechanism source:",
                        str(self._snapshot.source or "") or "(not provided)",
                    )
                ),
                is_error=True,
                base_executable_fingerprint=(
                    self._snapshot.base_executable_fingerprint
                ),
            )
        except Exception:
            logger.exception("Unexpected symbolic calculator evaluation failure")
            self._snapshot = None
            self._history.clear()
            self._available = False
            self._reason = "Symbolic calculator failed unexpectedly."
            return self.view_state()
        else:
            entry = SymbolicCalculatorHistoryEntry(
                query=query_s,
                result_text=str(result.result_text),
                context_text=str(result.full_context_copy_text()),
                is_error=False,
                base_executable_fingerprint=(
                    self._snapshot.base_executable_fingerprint
                ),
            )
        self._retain_latest_entry(entry)
        return self.view_state()

    def copy_compact_text(self) -> str:
        latest = self._copyable_entry()
        if latest is None:
            return ""
        return latest.result_text

    def copy_context_text(self) -> str:
        latest = self._copyable_entry()
        if latest is None:
            return ""
        return latest.context_text

    def view_state(self) -> SymbolicCalculatorViewState:
        latest = self._display_entry()
        can_copy = self._copyable_entry() is not None
        return SymbolicCalculatorViewState(
            available=bool(self._available),
            reason="" if self._available else str(self._reason or "Symbolic calculator unavailable."),
            latest_query="" if latest is None else latest.query,
            latest_result_text="" if latest is None else latest.result_text,
            latest_is_error=False if latest is None else bool(latest.is_error),
            can_copy_result=can_copy,
            can_copy_context=can_copy,
            equilibrium_summary=(
                ()
                if not self._available or self._snapshot is None
                else self._snapshot.equilibrium_summary
            ),
        )

    def _display_entry(self) -> SymbolicCalculatorHistoryEntry | None:
        # A historical entry is only current when its executable science is
        # the currently admitted calculator snapshot.  In particular, an
        # unavailable snapshot has no current result at all.
        if not self._available:
            return None
        return self._latest_entry_for_current_snapshot()

    def _copyable_entry(self) -> SymbolicCalculatorHistoryEntry | None:
        if not self._available:
            return None
        return self._latest_entry_for_current_snapshot()

    def _latest_entry_for_current_snapshot(self) -> SymbolicCalculatorHistoryEntry | None:
        if self._snapshot is None:
            return None
        fingerprint = self._snapshot.base_executable_fingerprint
        for entry in reversed(self._history):
            if entry.base_executable_fingerprint == fingerprint:
                return entry
        return None

    def _retain_latest_entry(self, entry: SymbolicCalculatorHistoryEntry) -> None:
        """Keep one latest evaluation for each accepted executable science."""
        self._history[:] = [
            retained
            for retained in self._history
            if retained.base_executable_fingerprint
            != entry.base_executable_fingerprint
        ]
        self._history.append(entry)

    def _build_snapshot(self) -> tuple[SymbolicCalculatorSnapshot | None, bool, str]:
        prepared_dsl = self._canonical_prepared_dsl()
        if prepared_dsl is None:
            return None, False, "Canonical mechanism is empty."
        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            raise TypeError(
                "Canonical symbolic input must be PreparedDslRequestSet or None."
            )
        source = self._canonical_source()
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("canonical_source must return MechanismAuthoringSource.")
        if not source_matches_analysis(source, prepared_dsl.analysis):
            return (
                None,
                False,
                "Canonical mechanism source and analysis no longer match.",
            )
        base_executable_fingerprint = (
            prepared_dsl.base_executable_identity().fingerprint
        )
        _require_executable_fingerprint(base_executable_fingerprint)
        try:
            parameterized_interventions = (
                prepared_dsl.parameterized_intervention_schedule_for_binding()
            )
            intervention_execution = (
                prepared_dsl.intervention_execution_contract()
            )
            if parameterized_interventions is not None or bool(
                intervention_execution is not None
                and not intervention_execution.is_empty
            ):
                return (
                    None,
                    False,
                    "Symbolic calculator is unavailable for mechanisms with "
                    "scheduled interventions.",
                )
            mechanism = prepared_dsl.mechanism_for_execution()
            apply_parameter_algebra_spec_to_mechanism(
                prepared_dsl.parameter_algebra_spec_for_execution(),
                mechanism=mechanism,
                require_mutable=False,
            )
            available, reason = symbolic_calculator_available(mechanism)
        except (DSLError, SymbolicCalculatorError) as exc:
            return None, False, str(exc) or "Symbolic calculator unavailable."
        except Exception:
            logger.exception("Unexpected symbolic calculator snapshot failure")
            return None, False, "Symbolic calculator failed unexpectedly."
        if not bool(available):
            return None, False, str(reason or "Symbolic calculator unavailable.")
        snapshot = SymbolicCalculatorSnapshot(
            source=source.reactions_text,
            mechanism=mechanism,
            equilibrium_summary=self._equilibrium_summary(mechanism),
            base_executable_fingerprint=base_executable_fingerprint,
        )
        return snapshot, True, ""

    @staticmethod
    def _equilibrium_summary(mechanism: object) -> tuple[str, ...]:
        equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
        temperature_K = MechanismMetadataView.from_metadata(
            getattr(mechanism, "metadata", {})
        ).temperature_K
        summaries: list[str] = []
        for index, equilibrium in enumerate(equilibria, start=1):
            contract = equilibrium_rate_contract(
                equilibrium,
                temperature_K=temperature_K,
            )
            summaries.append(_format_equilibrium_summary(index, contract))
        return tuple(summaries)


def _format_equilibrium_summary(
    index: int,
    contract: EquilibriumRateContract,
) -> str:
    """Render the canonical equilibrium DTO without re-deriving science."""

    return "; ".join(
        (
            f"Equilibrium {index}",
            _format_contract_value("Keq", contract.Keq, contract.Keq_unit),
            _format_contract_value("kf", contract.kf, contract.kf_unit),
            _format_contract_value("kr", contract.kr, contract.kr_unit),
        )
    )


def _format_contract_value(label: str, value: float | None, unit: str) -> str:
    if value is None:
        return f"{label} unavailable [{unit}]"
    return f"{label} = {value:.12g} [{unit}]"
