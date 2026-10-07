from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore

from kindred.core.mechanism_source import MechanismAuthoringSource, source_matches_analysis
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.errors import DSLError
from kindred.core.symbolic.calculator import SymbolicCalculatorError, symbolic_calculator_available
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.symbolic_calculator_worker import CalculatorWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorLatestResult:
    prepared_dsl: PreparedDslRequestSet
    query: str
    result_html: str
    context_text: str
    copy_text: str
    is_error: bool = False
    reuse_queries: tuple[tuple[str, str], ...] = ()


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
    latest_result_html: str = ""
    latest_is_error: bool = False
    can_copy_result: bool = False
    can_copy_context: bool = False
    busy: bool = False
    can_cancel: bool = False
    status: str = ""
    reuse_queries: tuple[tuple[str, str], ...] = ()


class SymbolicCalculatorOwner(QtCore.QObject):
    stateChanged = QtCore.Signal(object)
    application_close_settlement_changed = QtCore.Signal()
    disposed = QtCore.Signal(object)

    def __init__(self, *, canonical_prepared_dsl: Callable[[], PreparedDslRequestSet | None],
                 canonical_source: Callable[[], MechanismAuthoringSource], gate: ComputationGate, parent=None):
        super().__init__(parent)
        self._canonical_prepared_dsl = canonical_prepared_dsl
        self._canonical_source = canonical_source
        self._gate = gate
        self._snapshot: SymbolicCalculatorSnapshot | None = None
        self._latest_result: SymbolicCalculatorLatestResult | None = None
        self._worker: CalculatorWorker | None = None
        self._retiring: set[CalculatorWorker] = set()
        self._request: tuple[CalculatorWorker, SymbolicCalculatorSnapshot, str] | None = None
        self._available = False
        self._reason = "Canonical mechanism unavailable."
        self._status = ""
        self._closed = False
        self._disposing = False

    @property
    def busy(self) -> bool:
        return self._request is not None

    @property
    def application_close_settled(self) -> bool:
        return self._worker is None and not self._retiring

    def _publish(self) -> SymbolicCalculatorViewState:
        state = self.view_state()
        self.stateChanged.emit(state)
        self.application_close_settlement_changed.emit()
        return state

    def refresh(self) -> SymbolicCalculatorViewState:
        if self._closed:
            return self.view_state()
        previous = self._snapshot
        snapshot, available, reason = self._build_snapshot()
        if previous is not snapshot:
            self._retire_worker()
        self._snapshot, self._available, self._reason = snapshot, available, reason
        return self.view_state()

    def invalidate_current_snapshot(self, reason="Canonical mechanism has unapplied Reactions.") -> SymbolicCalculatorViewState:
        self._retire_worker()
        self._snapshot = None
        self._available = False
        self._reason = reason
        return self._publish()

    def _retire_worker(self) -> None:
        if self._worker is not None:
            worker, self._worker = self._worker, None
            self._retiring.add(worker)
            worker.request_shutdown()

    def cancel(self) -> SymbolicCalculatorViewState:
        self._status = "Canceling…" if self.busy else ""
        self._retire_worker()
        return self._publish()

    def cancel_for_application_close(self) -> None:
        self._closed = True
        self._available = False
        self._reason = "Calculator is closing."
        self.cancel()

    def dispose(self) -> None:
        self._disposing = True
        self.cancel_for_application_close()
        if self.application_close_settled:
            self.disposed.emit(self)

    def evaluate(self, query: str) -> SymbolicCalculatorViewState:
        query = query.strip()
        if self._closed or self.busy or self._retiring or not query:
            return self.view_state()
        self.refresh()
        if not self._available or self._snapshot is None:
            return self.view_state()
        if not self._gate.try_acquire(self, "calculator"):
            self._status = "Another computation is active."
            return self._publish()
        snapshot = self._snapshot
        if self._worker is None:
            worker = CalculatorWorker(snapshot.mechanism, snapshot.source, parent=self)
            self._worker = worker
            worker.result.connect(self._receive_result)
            worker.failed.connect(self._worker_failed)
            worker.finished.connect(self._worker_finished)
        worker = self._worker
        self._request = (worker, snapshot, query)
        self._status = f"Evaluating {query}"
        worker.submit(query)
        if not worker.isRunning():
            worker.start()
        return self._publish()

    @QtCore.Slot(str, str, str, bool, str, object)
    def _receive_result(self, query, result_html, context, error, copy_text, reuse_queries):
        worker = self.sender()
        if self._request is None or self._request[0] is not worker or self._worker is not worker:
            return
        _, snapshot, requested = self._request
        if self._closed or snapshot is not self._snapshot or requested != query:
            return
        self._latest_result = SymbolicCalculatorLatestResult(
            snapshot.prepared_dsl, query, result_html, context,
            copy_text, error, reuse_queries,
        )
        self._request = None
        self._status = ""
        self._gate.release(self)
        self._publish()

    @QtCore.Slot(str)
    def _worker_failed(self, message):
        if self.sender() is not self._worker:
            return
        self._status = message
        self._latest_result = None
        self._retire_worker()
        self._publish()

    @QtCore.Slot()
    def _worker_finished(self):
        worker = self.sender()
        self._retiring.discard(worker)
        if self._worker is worker:
            self._worker = None
        if self._request is not None and self._request[0] is worker:
            self._request = None
            if self._status.startswith("Canceling") or self._status.startswith("Evaluating"):
                self._status = "Canceled."
            self._gate.release(self)
        worker.deleteLater()
        self._publish()
        if self._disposing and self.application_close_settled:
            self.disposed.emit(self)

    def _current_result(self):
        if not self._available or self._snapshot is None or self._latest_result is None:
            return None
        if self._latest_result.prepared_dsl is not self._snapshot.prepared_dsl:
            return None
        return self._latest_result

    def copy_compact_text(self) -> str:
        latest = self._current_result()
        return latest.copy_text if latest is not None and not self.busy else ""

    def copy_context_text(self) -> str:
        latest = self._current_result()
        return latest.context_text if latest is not None and not self.busy else ""

    def reuse_query(self, displayed_query: str, index: int) -> str:
        if not self._available:
            return ""
        self.refresh()
        latest = self._current_result()
        if latest is None or self.busy or self._retiring or latest.query != displayed_query:
            return ""
        return latest.reuse_queries[index][1] if 0 <= index < len(latest.reuse_queries) else ""

    def view_state(self) -> SymbolicCalculatorViewState:
        latest = self._current_result()
        copyable = latest is not None and not self.busy
        return SymbolicCalculatorViewState(
            available=self._available and not self._closed,
            reason=self._reason if not self._available else "",
            latest_query=latest.query if latest else "",
            latest_result_html=latest.result_html if latest else "",
            latest_is_error=latest.is_error if latest else False,
            can_copy_result=copyable, can_copy_context=copyable,
            busy=self.busy or bool(self._retiring), can_cancel=self.busy and self._worker is not None,
            status=self._status,
            reuse_queries=latest.reuse_queries if copyable else (),
        )

    def _build_snapshot(self):
        try:
            prepared = self._canonical_prepared_dsl()
            if prepared is None:
                return None, False, "Canonical mechanism is empty."
            source = self._canonical_source()
            if self._snapshot is not None and self._snapshot.prepared_dsl is prepared and self._snapshot.source == source.reactions_text:
                return self._snapshot, True, ""
            if not source_matches_analysis(source, prepared.analysis):
                return None, False, "Canonical mechanism source and analysis no longer match."
            schedule = prepared.parameterized_intervention_schedule_for_binding()
            intervention = prepared.intervention_execution_contract()
            if schedule is not None or intervention is not None and not intervention.is_empty:
                return None, False, "Symbolic calculator is unavailable for scheduled interventions."
            mechanism = prepared.mutable_mechanism_shell()
            available, reason = symbolic_calculator_available(mechanism)
            if not available:
                return None, False, reason
            return SymbolicCalculatorSnapshot(source.reactions_text, mechanism, prepared), True, ""
        except (DSLError, SymbolicCalculatorError, ValueError) as exc:
            return None, False, str(exc)
        except Exception:
            logger.exception("Unexpected Calculator snapshot failure")
            return None, False, "Symbolic calculator failed unexpectedly."
