from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from kindred.core.mechanism_source import (
    MechanismAuthoringSource,
    source_matches_analysis,
)
from kindred.core.simulator.dsl_parse import DslAnalysis, PreparedDslRequestSet
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


def _calculator_input_value(
    prepared_dsl: PreparedDslRequestSet,
    source: MechanismAuthoringSource,
    mechanism: object,
) -> tuple[object, ...]:
    """Capture complete admitted calculator input as ordinary typed values."""

    reactions = tuple(
        (
            dict(getattr(item, "reactants", {}) or {}),
            dict(getattr(item, "products", {}) or {}),
            dict(getattr(item, "rate_orders", {}) or {}),
            dict(getattr(item, "overrides", {}) or {}),
            float(getattr(item, "rate")())
            if callable(getattr(item, "rate", None))
            else getattr(item, "rate", None),
        )
        for item in tuple(getattr(mechanism, "reactions", ()) or ())
    )
    equilibria = tuple(
        (
            dict(getattr(item, "stoich_forward", {}) or {}),
            dict(getattr(item, "stoich_back", {}) or {}),
            getattr(item, "Keq", None),
            getattr(item, "kf", None),
            getattr(item, "kr", None),
            dict(getattr(item, "metadata", {}) or {}),
        )
        for item in tuple(getattr(mechanism, "equilibria", ()) or ())
    )
    species = tuple(
        (str(name), float(getattr(item, "initial_conc", 0.0)))
        for name, item in dict(getattr(mechanism, "species", {}) or {}).items()
    )
    return (
        source,
        prepared_dsl.analysis,
        prepared_dsl.parameter_algebra_spec_for_execution(),
        species,
        reactions,
        equilibria,
    )


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorHistoryEntry:
    query: str
    result_text: str
    context_text: str
    admitted_input: object
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorSnapshot:
    source: str
    mechanism: object
    analysis: DslAnalysis
    admitted_input: object

@dataclass(frozen=True, slots=True)
class SymbolicCalculatorViewState:
    available: bool
    reason: str = ""
    latest_query: str = ""
    latest_result_text: str = ""
    latest_is_error: bool = False
    can_copy_result: bool = False
    can_copy_context: bool = False


class SymbolicCalculatorOwner:
    def __init__(
        self,
        *,
        canonical_prepared_dsl: Callable[[], PreparedDslRequestSet | None],
        canonical_source: Callable[[], MechanismAuthoringSource],
    ) -> None:
        self._canonical_prepared_dsl = canonical_prepared_dsl
        self._canonical_source = canonical_source
        self._snapshot: SymbolicCalculatorSnapshot | None = None
        self._history: list[SymbolicCalculatorHistoryEntry] = []
        self._available = False
        self._reason = "Canonical mechanism unavailable."

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
        preserving entries that may still match the accepted typed input after
        the draft is reverted or accepted.
        """

        self._snapshot = None
        self._available = False
        self._reason = str(reason).strip() or "Symbolic calculator unavailable."
        return self.view_state()

    def evaluate(self, query: str) -> SymbolicCalculatorViewState:
        query_s = str(query or "").strip()
        if not query_s:
            return self.view_state()
        # Accepted-source reconciliation explicitly refreshes this owner.  A
        # query must reuse the one target-local projection for that accepted
        # snapshot instead of rebuilding a mechanism shell for every query.
        if self._snapshot is None:
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
                admitted_input=self._snapshot.admitted_input,
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
                admitted_input=self._snapshot.admitted_input,
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
        for entry in reversed(self._history):
            if entry.admitted_input == self._snapshot.admitted_input:
                return entry
        return None

    def _retain_latest_entry(self, entry: SymbolicCalculatorHistoryEntry) -> None:
        """Keep one latest evaluation for each accepted typed calculator input."""
        self._history[:] = [
            retained
            for retained in self._history
            if retained.admitted_input != entry.admitted_input
        ]
        self._history.append(entry)

    def _build_snapshot(self) -> tuple[SymbolicCalculatorSnapshot | None, bool, str]:
        prepared_dsl = self._canonical_prepared_dsl()
        if prepared_dsl is None:
            return None, False, "Canonical mechanism is empty."
        source = self._canonical_source()
        if not source_matches_analysis(source, prepared_dsl.analysis):
            return (
                None,
                False,
                "Canonical mechanism source and analysis no longer match.",
            )
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
            mechanism = prepared_dsl.mutable_mechanism_shell()
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
            analysis=prepared_dsl.analysis,
            admitted_input=_calculator_input_value(
                prepared_dsl,
                source,
                mechanism,
            ),
        )
        return snapshot, True, ""
