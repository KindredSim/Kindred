"""Owner for transient Slider Preview execution.

The owner deliberately has no repository, batch identity, receipt, or plot
disposition route.  A frame is useful only while the one operation generation
is current; rendering errors remain local to the physical plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

from PySide6 import QtCore

from kindred.core.exceptions import SimulationCancelled
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulation_failure import FactualCause
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.interactive_preview import (
    InteractivePreviewFrame,
    InteractivePreviewRequest,
    InteractivePreviewSeries,
    InteractivePreviewTarget,
)
from kindred.gui.controllers.simulation_job_owner import (
    _captured_parameter_coordinates,
    _prepare_target,
    _simulation_worker_cause,
    _solve_prepared,
)


def _failure_message(cause: FactualCause | None, message: str) -> str:
    if cause is None:
        return str(message or "Interactive Preview failed.")
    return str(cause.message or message or "Interactive Preview failed.")


@dataclass(frozen=True, slots=True)
class _InteractiveTerminal:
    generation: int
    target: InteractivePreviewTarget
    state: str
    result: SimulationResultValue | None = None
    message: str = ""
    cause: FactualCause | None = None


class _InteractivePreviewWorker(QtCore.QThread):
    target_terminal = QtCore.Signal(int, object)

    def __init__(
        self,
        *,
        generation: int,
        request: InteractivePreviewRequest,
        cancellation_control: SimulationCancellationControl,
        cancel_event: Event,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._generation = int(generation)
        self._request = request
        self._control = cancellation_control
        self._cancel_event = cancel_event

    def cancel(self) -> None:
        self._cancel_event.set()
        self.requestInterruption()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        for target in self._request.targets:
            if self.cancelled:
                outcome = _InteractiveTerminal(
                    self._generation,
                    target,
                    "cancelled",
                    message="Interactive Preview cancelled before preparation.",
                )
            else:
                outcome = self._execute_target(target)
            self.target_terminal.emit(self._generation, outcome)
            if self.cancelled:
                break

    def _execute_target(self, target: InteractivePreviewTarget) -> _InteractiveTerminal:
        try:
            self._request.sampling.validate_execution_grid(
                target.execution_request.solver_config.grid
            )
            prepared, parameters = _prepare_target(
                target,  # structural target contract; interactive work has no repository key
                cancellation_control=self._control,
            )
            result = _solve_prepared(
                prepared,
                parameters=parameters,
                cancellation_control=self._control,
            )
        except SimulationCancelled:
            return _InteractiveTerminal(
                self._generation,
                target,
                "cancelled",
                message="Interactive Preview cancelled.",
            )
        except BaseException as exc:  # noqa: BLE001
            return _InteractiveTerminal(
                self._generation,
                target,
                "failure",
                message=str(exc) or type(exc).__name__,
                cause=_simulation_worker_cause(exc),
            )
        return _InteractiveTerminal(
            self._generation,
            target,
            "success",
            result=result,
        )


class InteractivePreviewOwner(QtCore.QObject):
    """Own one cancellable, transient Slider Preview operation."""

    state_changed = QtCore.Signal()
    frame_ready = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    settled_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        gate: ComputationGate,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(gate, ComputationGate):
            raise TypeError("gate must be a ComputationGate.")
        self._gate = gate
        self._generation = 0
        self._current_request: InteractivePreviewRequest | None = None
        self._pending_request: InteractivePreviewRequest | None = None
        self._worker: _InteractivePreviewWorker | None = None
        self._cancel_event: Event | None = None
        self._cancellation_control: SimulationCancellationControl | None = None
        self._terminal_state: str | None = None
        self._close_fenced = False
        self._last_settled = True
        self._gate.changed.connect(self._on_gate_changed)

    @property
    def busy(self) -> bool:
        return bool(self._worker is not None or self._pending_request is not None)

    @property
    def current_kind(self) -> str | None:
        return "interactive" if self.busy else None

    @property
    def application_close_settled(self) -> bool:
        return bool(
            self._worker is None
            and self._pending_request is None
            and self._current_request is None
        )

    def _mark_unsettled(self) -> None:
        self._last_settled = False

    def _emit_settled_if_complete(self) -> None:
        if not self._last_settled and self.application_close_settled:
            self._last_settled = True
            self.settled_changed.emit()

    def request(self, request: InteractivePreviewRequest) -> bool:
        if not isinstance(request, InteractivePreviewRequest):
            raise TypeError("request must be InteractivePreviewRequest.")
        if self._close_fenced:
            self.status_changed.emit("Interactive Preview rejected: owner is closing.")
            return False
        self._generation += 1
        self._pending_request = request
        self._mark_unsettled()
        if self._worker is not None:
            self._worker.cancel()
            self.state_changed.emit()
            return False
        return self._start_pending()

    def _start_pending(self) -> bool:
        request = self._pending_request
        if request is None or self._worker is not None or not self._gate.is_idle:
            return False
        if self._close_fenced:
            self._pending_request = None
            self._emit_settled_if_complete()
            return False
        if not self._gate.try_acquire(self, "preview"):
            return False
        self._pending_request = None
        self._current_request = request
        self._terminal_state = None
        cancel_event = Event()
        control = SimulationCancellationControl(cancel_event.is_set)
        self._cancel_event = cancel_event
        self._cancellation_control = control
        generation = self._generation
        worker = _InteractivePreviewWorker(
            generation=generation,
            request=request,
            cancellation_control=control,
            cancel_event=cancel_event,
            parent=self,
        )
        self._worker = worker
        worker.target_terminal.connect(self._on_target_terminal)
        worker.finished.connect(self._on_worker_finished)
        try:
            worker.start()
        except BaseException as exc:  # noqa: BLE001
            self._terminal_state = "failure"
            self._worker = None
            self._current_request = None
            self._cancellation_control = None
            self._cancel_event = None
            self._gate.release(self)
            self.status_changed.emit(f"Interactive Preview dispatch failed: {exc}")
            self.state_changed.emit()
            self._emit_settled_if_complete()
            return False
        self.state_changed.emit()
        return True

    def _is_current(self, generation: int) -> bool:
        request = self._current_request
        return bool(
            self._worker is not None
            and request is not None
            and int(generation) == self._generation
        )

    @QtCore.Slot(int, object)
    def _on_target_terminal(self, generation: int, outcome: object) -> None:
        if not isinstance(outcome, _InteractiveTerminal) or not self._is_current(generation):
            return
        target = outcome.target
        if outcome.state != "success" or not isinstance(outcome.result, SimulationResultValue):
            self._terminal_state = outcome.state
            if outcome.state == "failure":
                self.status_changed.emit(_failure_message(outcome.cause, outcome.message))
            return
        result = outcome.result
        frame = InteractivePreviewFrame(
            tab_id=self._current_request.tab_id,
            set_id=target.set_id,
            generation=generation,
            target_set_ids=self._current_request.target_set_ids,
            x_values=tuple(float(value) for value in result.t),
            segment_ids=tuple(int(value) for value in result.segment_ids),
            segment_keys=tuple(result.segment_keys),
            discontinuity_ids=tuple(int(value) for value in result.discontinuity_ids),
            presentation_boundaries=result.presentation_boundaries,
            series=tuple(
                InteractivePreviewSeries(
                    series_id=name,
                    label=name,
                    values=tuple(float(value) for value in values),
                )
                for name, values in result.series.items()
            ),
        )
        # The sole currentness comparison is immediately before publication.
        self.frame_ready.emit(frame)

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return
        self._worker = None
        self._current_request = None
        self._cancellation_control = None
        self._cancel_event = None
        self._terminal_state = self._terminal_state or "cancelled"
        self._gate.release(self)
        if worker is not None:
            worker.deleteLater()
        self.state_changed.emit()
        self._emit_settled_if_complete()
        if not self._close_fenced:
            QtCore.QTimer.singleShot(0, self._start_pending)

    def retire(
        self,
        tab_id: str | None = None,
        set_ids: tuple[str, ...] | None = None,
    ) -> bool:
        tab = None if tab_id is None else str(tab_id).strip()
        affected = None if set_ids is None else {str(value) for value in set_ids}
        changed = False
        if (
            self._pending_request is not None
            and (tab is None or self._pending_request.tab_id == tab)
            and (
                affected is None
                or affected.intersection(self._pending_request.target_set_ids)
            )
        ):
            self._pending_request = None
            changed = True
        if (
            self._current_request is not None
            and (tab is None or self._current_request.tab_id == tab)
            and (
                affected is None
                or affected.intersection(self._current_request.target_set_ids)
            )
        ):
            self._generation += 1
            if self._worker is not None:
                self._worker.cancel()
            changed = True
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def cancel_current(self) -> bool:
        return self.retire()

    def fence_application_close_admission(self) -> None:
        self._close_fenced = True
        self.retire()

    def cancel_for_application_close(self) -> None:
        self.fence_application_close_admission()

    def request_project_replacement_cancellation(self) -> None:
        self.retire()

    def settle_project_replacement(self, deadline: float) -> bool:
        while not self.application_close_settled:
            if self._worker is None:
                self._start_pending()
                break
            remaining = max(0.0, float(deadline) - __import__("time").monotonic())
            if remaining <= 0.0:
                return False
            self._worker.wait(int(min(remaining, 0.05) * 1000))
            QtCore.QCoreApplication.processEvents()
        return self.application_close_settled

    @QtCore.Slot()
    def _on_gate_changed(self) -> None:
        if self._gate.is_idle and self._worker is None and self._pending_request is not None:
            QtCore.QTimer.singleShot(0, self._start_pending)
