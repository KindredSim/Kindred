from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from kindred.core.batch_initial_conditions import strip_named_reaction_dsl_initial_concentration_sets
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.symbolic.calculator import (
    SymbolicCalculatorError,
    evaluate_symbolic_query,
    symbolic_calculator_available,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorHistoryEntry:
    snapshot_source: str
    query: str
    result_text: str
    context_text: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorSnapshot:
    source: str
    parse_source: str
    mechanism: Any


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorViewState:
    available: bool
    reason: str = ""
    latest_query: str = ""
    latest_result_text: str = ""
    latest_context_text: str = ""
    latest_is_error: bool = False
    history_count: int = 0
    can_copy_result: bool = False
    can_copy_context: bool = False


class SymbolicCalculatorOwner:
    def __init__(
        self,
        *,
        mechanism_session_owner: Any,
    ) -> None:
        self._mechanism_session_owner = mechanism_session_owner
        self._snapshot: SymbolicCalculatorSnapshot | None = None
        self._history: list[SymbolicCalculatorHistoryEntry] = []
        self._available = False
        self._reason = "Canonical mechanism unavailable."
        self._hide_history_while_unavailable = False

    def refresh(self) -> SymbolicCalculatorViewState:
        self._snapshot, self._available, self._reason = self._build_snapshot()
        if self._available:
            self._hide_history_while_unavailable = False
        elif self._reason == "Symbolic calculator failed unexpectedly.":
            self._history.clear()
            self._hide_history_while_unavailable = True
        return self.view_state()

    def reset_project_session_state(self) -> SymbolicCalculatorViewState:
        self._snapshot = None
        self._history.clear()
        self._available = False
        self._reason = "Canonical mechanism unavailable."
        self._hide_history_while_unavailable = False
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
                snapshot_source=self._snapshot.source,
                query=query_s,
                result_text=message,
                context_text=f"Query: {query_s}\nError: {message}",
                is_error=True,
            )
        except Exception:
            logger.exception("Unexpected symbolic calculator evaluation failure")
            self._snapshot = None
            self._history.clear()
            self._available = False
            self._reason = "Symbolic calculator failed unexpectedly."
            self._hide_history_while_unavailable = True
            return self.view_state()
        else:
            entry = SymbolicCalculatorHistoryEntry(
                snapshot_source=self._snapshot.source,
                query=query_s,
                result_text=str(result.result_text),
                context_text=str(result.full_context_copy_text()),
                is_error=False,
            )
        self._history.append(entry)
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
            latest_context_text="" if latest is None else latest.context_text,
            latest_is_error=False if latest is None else bool(latest.is_error),
            history_count=len(self._history),
            can_copy_result=can_copy,
            can_copy_context=can_copy,
        )

    def _display_entry(self) -> SymbolicCalculatorHistoryEntry | None:
        if self._available:
            return self._latest_entry_for_current_snapshot()
        if bool(getattr(self, "_hide_history_while_unavailable", False)):
            return None
        return self._history[-1] if self._history else None

    def _copyable_entry(self) -> SymbolicCalculatorHistoryEntry | None:
        if not self._available:
            return None
        return self._latest_entry_for_current_snapshot()

    def _latest_entry_for_current_snapshot(self) -> SymbolicCalculatorHistoryEntry | None:
        if self._snapshot is None:
            return None
        source = self._snapshot.source
        for entry in reversed(self._history):
            if entry.snapshot_source == source:
                return entry
        return None

    def _build_snapshot(self) -> tuple[SymbolicCalculatorSnapshot | None, bool, str]:
        owner = self._mechanism_session_owner
        if owner is None:
            return None, False, "Canonical mechanism unavailable."
        if bool(getattr(owner, "edit_session_active", False)):
            return None, False, "Symbolic calculator is disabled while mechanism editing is active."
        validate = getattr(owner, "validate_canonical", None)
        if not callable(validate):
            return None, False, "Canonical mechanism validation is unavailable."
        validation = validate()
        if not bool(getattr(validation, "valid", False)):
            reason = str(getattr(validation, "error_message", "") or "Canonical mechanism is invalid.")
            return None, False, reason
        ready_for_run = getattr(owner, "is_ready_for_explicit_run", None)
        if not callable(ready_for_run):
            return None, False, "Canonical mechanism run readiness is unavailable."
        if not bool(ready_for_run()):
            return None, False, "Canonical mechanism is not ready for explicit runs."
        source = str(getattr(owner, "canonical_full_dsl", "") or "")
        try:
            parse_source = str(strip_named_reaction_dsl_initial_concentration_sets(source) or "")
            mechanism = parse_dsl_to_mechanism(parse_source, initials={})
            available, reason = symbolic_calculator_available(mechanism)
        except (DSLError, SymbolicCalculatorError) as exc:
            return None, False, str(exc) or "Symbolic calculator unavailable."
        except Exception:
            logger.exception("Unexpected symbolic calculator snapshot failure")
            return None, False, "Symbolic calculator failed unexpectedly."
        if not bool(available):
            return None, False, str(reason or "Symbolic calculator unavailable.")
        snapshot = SymbolicCalculatorSnapshot(
            source=source,
            parse_source=parse_source,
            mechanism=mechanism,
        )
        return snapshot, True, ""
