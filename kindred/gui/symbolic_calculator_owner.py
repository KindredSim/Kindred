from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from kindred.core.mechanism_source import (
    MechanismAuthoringSource,
    source_matches_analysis,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.simulation_preparation import _build_parameter_target_map
from kindred.core.symbolic.calculator import (
    SymbolicCalculatorError,
    evaluate_symbolic_query,
    symbolic_calculator_available,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorLatestResult:
    prepared_dsl: PreparedDslRequestSet
    query: str
    result_text: str
    context_text: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorSnapshot:
    source: str
    mechanism: object
    prepared_dsl: PreparedDslRequestSet


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
        self._latest_result: SymbolicCalculatorLatestResult | None = None
        self._available = False
        self._reason = "Canonical mechanism unavailable."

    def refresh(self) -> SymbolicCalculatorViewState:
        self._snapshot, self._available, self._reason = self._build_snapshot()
        if (
            not self._available
            and self._reason == "Symbolic calculator failed unexpectedly."
        ):
            self._latest_result = None
        return self.view_state()

    def invalidate_current_snapshot(
        self,
        reason: str = "Canonical mechanism has unapplied Reactions.",
    ) -> SymbolicCalculatorViewState:
        """Hide the current result while retaining it for an accepted revert.

        Reactions drafts are a temporary authoring state.  Clearing the current
        snapshot makes display and copy unavailable until the accepted DSL is
        rebuilt, while retaining the one result that may match after a draft is
        reverted without changing the accepted prepared DSL.
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
        snapshot = self._snapshot
        try:
            result = evaluate_symbolic_query(
                snapshot.mechanism,
                query_s,
                mechanism_source=snapshot.source,
            )
        except SymbolicCalculatorError as exc:
            message = str(exc)
            latest = SymbolicCalculatorLatestResult(
                prepared_dsl=snapshot.prepared_dsl,
                query=query_s,
                result_text=message,
                context_text="\n".join(
                    (
                        f"Query: {query_s}",
                        f"Error: {message}",
                        "Mechanism source:",
                        str(snapshot.source or "") or "(not provided)",
                    )
                ),
                is_error=True,
            )
        except Exception:
            logger.exception("Unexpected symbolic calculator evaluation failure")
            self._snapshot = None
            self._latest_result = None
            self._available = False
            self._reason = "Symbolic calculator failed unexpectedly."
            return self.view_state()
        else:
            latest = SymbolicCalculatorLatestResult(
                prepared_dsl=snapshot.prepared_dsl,
                query=query_s,
                result_text=str(result.result_text),
                context_text=str(result.full_context_copy_text()),
                is_error=False,
            )
        self._latest_result = latest
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
            reason=""
            if self._available
            else str(self._reason or "Symbolic calculator unavailable."),
            latest_query="" if latest is None else latest.query,
            latest_result_text="" if latest is None else latest.result_text,
            latest_is_error=False if latest is None else bool(latest.is_error),
            can_copy_result=can_copy,
            can_copy_context=can_copy,
        )

    def _display_entry(self) -> SymbolicCalculatorLatestResult | None:
        # A result is current only when its prepared DSL is the currently
        # admitted calculator snapshot.  An unavailable snapshot has no result.
        if not self._available:
            return None
        return self._latest_entry_for_current_snapshot()

    def _copyable_entry(self) -> SymbolicCalculatorLatestResult | None:
        if not self._available:
            return None
        return self._latest_entry_for_current_snapshot()

    def _latest_entry_for_current_snapshot(
        self,
    ) -> SymbolicCalculatorLatestResult | None:
        if self._snapshot is None or self._latest_result is None:
            return None
        if self._latest_result.prepared_dsl is not self._snapshot.prepared_dsl:
            return None
        return self._latest_result

    def _build_snapshot(self) -> tuple[SymbolicCalculatorSnapshot | None, bool, str]:
        try:
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
            parameterized_interventions = (
                prepared_dsl.parameterized_intervention_schedule_for_binding()
            )
            intervention_execution = prepared_dsl.intervention_execution_contract()
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
            parameter_target_map = _build_parameter_target_map(mechanism)
            apply_parameter_algebra_spec_to_mechanism(
                prepared_dsl.parameter_algebra_spec_for_execution(),
                mechanism=mechanism,
                require_mutable=False,
                parameter_target_map=parameter_target_map,
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
            prepared_dsl=prepared_dsl,
        )
        return snapshot, True, ""
