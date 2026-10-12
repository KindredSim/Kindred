"""Direct Run and exact Preview lifecycle owner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from threading import Event
from typing import Callable, Iterable, Literal, Mapping

from PySide6 import QtCore

from kindred.core.exceptions import SimulationCancelled
from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
)
from kindred.core.result_repository import ResultRepository, SimulationResultValue
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    build_simulation_execution_request,
)
from kindred.core.simulation_execution import execute_simulation
from kindred.core.simulation_preparation_stages import build_solver_config
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.interactive_preview import (
    InteractivePreviewRequest,
    InteractivePreviewTarget,
)
from kindred.gui.workspace_tab_registry import ScientificDocumentDelta

__all__ = [
    "ExplicitRunCapture",
    "PreviewCapture",
    "SimulationJobOwner",
    "build_explicit_target",
    "build_execution_request",
]


def resolve_run_parallel_capacity(*, cpu_count: object, user_max: object, target_count: int) -> int:
    cpu = int(cpu_count) if type(cpu_count) is int and cpu_count > 0 else 1
    budget = max(1, cpu - 1)
    default = min(budget, 16)
    resolved = max(1, min(int(user_max), MAX_PARALLEL_WORKERS_CEILING)) if type(user_max) is int else default
    return min(target_count, budget, resolved, MAX_PARALLEL_WORKERS_CEILING)


@dataclass(frozen=True, slots=True)
class ExplicitRunTarget:
    """One Run target, built from document identities without Preview values."""

    set_id: str
    set_name: str
    execution_request: SimulationExecutionRequest


@dataclass(frozen=True, slots=True)
class ExplicitRunCapture:
    tab_id: str
    targets: tuple[ExplicitRunTarget, ...]


@dataclass(frozen=True, slots=True)
class PreviewTarget:
    set_id: str
    set_name: str
    execution_request: SimulationExecutionRequest
    canonical_run_target: ExplicitRunTarget


@dataclass(frozen=True, slots=True)
class PreviewCapture:
    """One exact Preview action."""

    tab_id: str
    targets: tuple[PreviewTarget, ...]


@dataclass(frozen=True, slots=True)
class SimulationTerminalSummary:
    kind: str
    state: str
    succeeded: int
    failed: int
    cancelled: int
    discarded: int
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TargetTerminalOutcome:
    target: ExplicitRunTarget | PreviewTarget | InteractivePreviewTarget
    state: Literal["success", "failure", "cancelled"]
    result: SimulationResultValue | None = None
    message: str = ""


def build_execution_request(
    *,
    prepared_dsl: PreparedDslRequestSet,
    initial_values: Iterable[tuple[str, float]],
    execution_settings: SimulationExecutionSettings,
    scientific_settings: TabScientificSettings,
    point_count: int | None = None,
    parameter_values: Mapping[str, float] | None = None,
) -> SimulationExecutionRequest:
    """Assemble Run or Preview science from its owner's captured inputs."""
    solver_config = build_solver_config(
        solver=execution_settings.solver,
        rtol=scientific_settings.rtol,
        atol=scientific_settings.atol,
        grid={"N": execution_settings.integration_points if point_count is None else point_count},
        use_analytic_jacobian=scientific_settings.analytic_jacobian_policy,
    )
    return build_simulation_execution_request(
        prepared_dsl=prepared_dsl,
        initial_values=initial_values,
        t_span=(execution_settings.t_start, execution_settings.t_end),
        solver_config=solver_config,
        parameter_values=parameter_values,
        use_authored_stops=scientific_settings.use_authored_stops,
    )


def build_explicit_target(*, prepared_dsl: PreparedDslRequestSet, set_id: str, set_name: str,
                          initial_values: Iterable[tuple[str, float]],
                          execution_settings: SimulationExecutionSettings, scientific_settings: TabScientificSettings) -> ExplicitRunTarget:
    execution_request = build_execution_request(
        prepared_dsl=prepared_dsl,
        initial_values=initial_values,
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
    )
    return ExplicitRunTarget(set_id, set_name, execution_request)


class _DirectSimulationWorker(QtCore.QThread):
    target_terminal = QtCore.Signal(int, object)

    def __init__(
        self,
        *,
        generation: int,
        targets: tuple[
            ExplicitRunTarget | PreviewTarget | InteractivePreviewTarget,
            ...,
        ],
        max_parallel: int,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._generation = int(generation)
        self._targets = tuple(targets)
        self._max_parallel = max(1, min(int(max_parallel), len(self._targets)))
        self._cancel_event = Event()
        self._control = SimulationCancellationControl(self._cancel_event.is_set)

    def cancel(self) -> None:
        self._cancel_event.set()
        self.requestInterruption()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        if self.cancelled:
            for target in self._targets:
                self.target_terminal.emit(
                    self._generation,
                    _TargetTerminalOutcome(
                        target=target,
                        state="cancelled",
                        message="Simulation canceled.",
                    ),
                )
            return
        with ThreadPoolExecutor(max_workers=self._max_parallel, thread_name_prefix="kindred-run") as pool:
            futures = {pool.submit(self._execute_target, target): target for target in self._targets}
            for future in as_completed(futures):
                target = futures[future]
                try:
                    outcome = future.result()
                except BaseException as exc:  # noqa: BLE001
                    outcome = _TargetTerminalOutcome(
                        target=target, state="cancelled" if self.cancelled else "failure",
                        message=str(exc),
                    )
                self.target_terminal.emit(self._generation, outcome)
                if self.cancelled:
                    for pending in futures:
                        pending.cancel()

    def _execute_target(
        self,
        target: ExplicitRunTarget | PreviewTarget | InteractivePreviewTarget,
    ) -> _TargetTerminalOutcome:
        if self.cancelled:
            return _TargetTerminalOutcome(target, "cancelled", message="Simulation canceled.")
        try:
            result = execute_simulation(
                target.execution_request, cancellation_control=self._control,
            )
        except SimulationCancelled:
            return _TargetTerminalOutcome(target, "cancelled", message="Simulation canceled.")
        except BaseException as exc:  # noqa: BLE001
            return _TargetTerminalOutcome(target, "failure", message=str(exc))
        return _TargetTerminalOutcome(target, "success", result=result)


RepositoryResolver = Callable[[str], ResultRepository | None]
RawUserParallelMaximum = Callable[[], object]
class SimulationJobOwner(QtCore.QObject):
    """Own Run, exact Preview, and interactive Preview computation."""

    state_changed = QtCore.Signal()
    application_close_settlement_changed = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    results_changed = QtCore.Signal(str)
    run_dispatched = QtCore.Signal(str, object)
    interactive_result_ready = QtCore.Signal(str, str, object, object)
    interactive_target_failed = QtCore.Signal(str, str, object)

    def __init__(self, *, gate: ComputationGate, repository_for_tab: RepositoryResolver,
                 user_parallel_maximum: RawUserParallelMaximum,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._gate = gate
        self._repository_for_tab = repository_for_tab
        self._user_parallel_maximum = user_parallel_maximum
        self._generation = 0
        self._operation: dict[str, object] | None = None
        self._worker: _DirectSimulationWorker | None = None
        self._terminal_state: str | None = None
        self._pending_preview: PreviewCapture | None = None
        self._pending_interactive_preview: InteractivePreviewRequest | None = None
        self._close_fenced = False
        self._last_close_settled = True
        # Latest failed canonical Run request per (tab, set); a Reference Run
        # exactly equal to it is not requested again.
        self._failed_run_requests: dict[tuple[str, str], SimulationExecutionRequest] = {}
        self._gate.changed.connect(self._on_gate_changed)

    @property
    def application_close_settled(self) -> bool:
        return bool(
            self._worker is None
            and self._operation is None
            and self._pending_preview is None
            and self._pending_interactive_preview is None
        )

    @property
    def busy(self) -> bool:
        return bool(
            self._worker is not None
            or self._operation is not None
            or self._pending_preview is not None
            or self._pending_interactive_preview is not None
        )

    @property
    def interactive_preview_busy(self) -> bool:
        return bool(
            self._pending_interactive_preview is not None
            or (
                self._operation is not None
                and self._operation["kind"] == "preview"
                and self._operation["purpose"] == "interactive-preview"
            )
        )

    @property
    def current_kind(self) -> str | None:
        if self._operation is not None:
            return str(self._operation["kind"])
        if self._pending_interactive_preview is not None:
            return "preview"
        return "preview" if self._pending_preview is not None else None

    @property
    def origin_tab_id(self) -> str | None:
        operation = self._operation
        if operation is not None:
            return str(operation["tab_id"])
        if self._pending_interactive_preview is not None:
            return self._pending_interactive_preview.tab_id
        return None if self._pending_preview is None else self._pending_preview.tab_id

    @property
    def pending_preview_tab_id(self) -> str | None:
        return None if self._pending_preview is None else self._pending_preview.tab_id

    def preview_set_ids(self, tab_id: str) -> tuple[str, ...]:
        """Return semantic exact Preview identities for one tab."""

        return self._repository(tab_id).preview_set_ids()

    def owns_preview_capture(self, capture: PreviewCapture) -> bool:
        """Report whether an exact Preview capture still owns live work."""

        if self._pending_preview is capture:
            return True

        operation = self._operation
        if (
            operation is not None
            and str(operation["kind"]) == "preview"
            and str(operation["purpose"]) == "ordinary"
            and self._operation_is_current(int(operation["generation"]))
        ):
            operation_targets = tuple(operation["targets"])
            if len(operation_targets) == len(capture.targets) and all(
                operation_target is capture_target
                for operation_target, capture_target in zip(
                    operation_targets,
                    capture.targets,
                    strict=True,
                )
            ):
                return True

        repository = self._repository(capture.tab_id)
        capture_ids = tuple(target.set_id for target in capture.targets)
        if set(repository.preview_set_ids()) != set(capture_ids):
            return False
        return all(
            repository.current_request(target.set_id)
            is target.execution_request
            and repository.current_result(target.set_id) is not None
            for target in capture.targets
        )

    def _mark_unsettled(self) -> None:
        self._last_close_settled = False

    def _emit_settled_if_complete(self) -> None:
        if not self._last_close_settled and self.application_close_settled:
            self._last_close_settled = True
            self.application_close_settlement_changed.emit()

    def _repository(self, tab_id: str) -> ResultRepository:
        return self._repository_for_tab(tab_id)

    def _operation_is_current(self, generation: int) -> bool:
        operation = self._operation
        return bool(self._worker is not None and operation is not None and int(operation["generation"]) == self._generation == int(generation))

    def _retire_if_affected(
        self,
        tab_id: str,
        set_ids: Iterable[str],
        *,
        include_run: bool = True,
    ) -> bool:
        tab = tab_id
        ids = set(set_ids)
        if not ids:
            return False
        operation = self._operation
        if (
            operation is None
            or not self._operation_is_current(operation["generation"])
            or operation["tab_id"] != tab
        ):
            return False
        if not include_run and operation["kind"] != "preview":
            return False
        if not ids.intersection(target.set_id for target in operation["targets"]):
            return False
        self._generation += 1
        self._worker.cancel()
        self._terminal_state = "cancelled"
        return True

    def reset_project_session(self) -> None:
        """Forget failed-Run memory when an actual project replacement occurs."""

        self._failed_run_requests.clear()

    def _retire_unserved_reference_run(self) -> None:
        """A Reference Run lives only while its pending exact Preview needs it."""

        operation = self._operation
        if operation is None or operation["purpose"] != "preview-reference":
            return
        pending = self._pending_preview
        if pending is not None and pending.tab_id == operation["tab_id"] and any(
            run_target.set_id == target.set_id
            and (run_target.execution_request == target.canonical_run_target.execution_request)
            for run_target in operation["targets"]
            for target in pending.targets
        ):
            return
        self._retire_if_affected(
            operation["tab_id"],
            (target.set_id for target in operation["targets"]),
        )

    def _retire_pending_preview_if_affected(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> bool:
        pending = self._pending_preview
        if pending is None or pending.tab_id != str(tab_id):
            return False
        ids = set(set_ids)
        if not ids.intersection(target.set_id for target in pending.targets):
            return False
        self._pending_preview = None
        return True

    def fence_document_delta(self, delta: ScientificDocumentDelta) -> None:
        """Fence computation for one admitted document delta.

        The registry owns repository mutation and physical presentation
        retirement.  This operation only retires computation intent and
        records removed identities on the operation being cancelled.
        """

        targets_by_tab = delta.targets_by_tab()
        for tab, set_ids in delta.removed_targets_by_tab:
            for set_id in set_ids:
                self._failed_run_requests.pop((tab, set_id), None)
        state_changed = False
        operation = self._operation
        if operation is not None:
            operation_tab = operation["tab_id"]
            operation["removed_target_ids"].update(
                dict(delta.removed_targets_by_tab).get(operation_tab, ())
            )
            state_changed = self._retire_if_affected(
                operation_tab, targets_by_tab.get(operation_tab, ()),
            )
        pending_preview = self._pending_preview
        if pending_preview is not None and set(
            targets_by_tab.get(pending_preview.tab_id, ())
        ).intersection(target.set_id for target in pending_preview.targets):
            self._pending_preview = None
            state_changed = True
        pending_interactive = self._pending_interactive_preview
        if pending_interactive is not None and set(
            targets_by_tab.get(pending_interactive.tab_id, ())
        ).intersection(pending_interactive.target_set_ids):
            self._pending_interactive_preview = None
            state_changed = True
        self._retire_unserved_reference_run()
        if state_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()

    def reconcile_canonical_targets(self, tab_id: str, set_ids: Iterable[str], *, clear_pending: bool = True) -> tuple[tuple[str, bool], ...]:
        ids = tuple(dict.fromkeys(set_ids))
        operation = self._operation
        if not (
            operation is not None
            and str(operation["kind"]) == "preview"
            and str(operation["purpose"]) == "interactive-preview"
        ):
            self._retire_if_affected(tab_id, ids, include_run=False)
        interactive_changed = self.retire_interactive_preview(tab_id, ids)
        pending_changed = clear_pending and self._retire_pending_preview_if_affected(tab_id, ids)
        self._retire_unserved_reference_run()
        repository = self._repository(tab_id)
        restored: list[tuple[str, bool]] = []
        for set_id in ids:
            repository.clear_preview(set_id, clear_reference=False)
            restored_value = repository.restore_reference_or_current(set_id)
            restored.append((set_id, restored_value))
        self.results_changed.emit(str(tab_id))
        if pending_changed and not interactive_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return tuple(restored)

    def start_run(self, capture: ExplicitRunCapture) -> bool:
        if self._close_fenced or self._worker is not None or not self._gate.is_idle:
            self.status_changed.emit("Run rejected: another computation is active.")
            return False
        return self._start(
            capture.tab_id,
            "run",
            capture.targets,
            max_parallel=self._capacity(len(capture.targets)),
        )

    def _capacity(self, count: int) -> int:
        return resolve_run_parallel_capacity(
            cpu_count=os.cpu_count(),
            user_max=self._user_parallel_maximum(),
            target_count=count,
        )

    def _start(
        self,
        tab_id: str,
        kind: str,
        targets: tuple[
            ExplicitRunTarget | PreviewTarget | InteractivePreviewTarget,
            ...,
        ],
        *,
        max_parallel: int,
        purpose: str = "ordinary",
        payload: object | None = None,
    ) -> bool:
        if self._worker is not None or not targets:
            return False
        if not self._gate.try_acquire(self, kind):
            return False
        self._mark_unsettled()
        self._generation += 1
        generation = self._generation
        worker = _DirectSimulationWorker(generation=generation, targets=targets, max_parallel=max_parallel, parent=self)
        self._operation = {
            "generation": generation,
            "kind": kind,
            "purpose": purpose,
            "tab_id": str(tab_id),
            "targets": targets,
            "outcomes": [],
            "details": [],
            "payload": payload,
            "preview_results": {},
            "removed_target_ids": set(),
        }
        self._worker = worker
        self._terminal_state = None
        worker.target_terminal.connect(self._on_target_terminal)
        worker.finished.connect(self._on_worker_finished)
        worker.start()
        if kind == "run" and purpose == "ordinary":
            repository = self._repository(str(tab_id))
            changed_request_ids = tuple(
                target.set_id
                for target in targets
                if not self._current_run_matches_target(repository, target)
            )
            if changed_request_ids:
                repository.deauthorize_sets(changed_request_ids)
                self.results_changed.emit(str(tab_id))
        self.state_changed.emit()
        if kind == "run":
            self.run_dispatched.emit(str(tab_id), tuple(target.set_id for target in targets))
        return True

    @staticmethod
    def _current_run_matches_target(
        repository: ResultRepository,
        target: ExplicitRunTarget,
    ) -> bool:
        """Keep a current result only for the exact ordinary Run request."""

        current_request = repository.current_request(target.set_id)
        current_result = repository.current_result(target.set_id)
        return bool(
            current_request is not None
            and current_result is not None
            and (current_request == target.execution_request)
        )

    def capture_settled_preview(
        self,
        tab_id: str,
        roster_set_ids: tuple[str, ...],
        *,
        expected_requests: tuple[tuple[str, SimulationExecutionRequest], ...],
    ) -> tuple[
        tuple[str, SimulationExecutionRequest, SimulationResultValue], ...
    ] | None:
        if self.busy or not self._gate.is_idle:
            return None
        repository = self._repository(tab_id)
        if set(repository.preview_set_ids()) != set(roster_set_ids):
            return None
        expected = dict(expected_requests)
        captured: list[
            tuple[str, SimulationExecutionRequest, SimulationResultValue]
        ] = []
        for set_id in roster_set_ids:
            current_request = repository.current_request(set_id)
            result = repository.current_result(set_id)
            if (
                current_request is None
                or result is None
                or expected.get(str(set_id)) is not current_request
            ):
                return None
            captured.append((set_id, current_request, result))
        return tuple(captured)

    def promote_preview_to_run(
        self,
        tab_id: str,
        preview_pairs: tuple[
            tuple[str, SimulationExecutionRequest, SimulationResultValue], ...
        ],
    ) -> None:
        repository = self._repository(tab_id)
        for set_id, request, result in preview_pairs:
            repository.publish_run(set_id, request, result)
        self.results_changed.emit(str(tab_id))

    def request_preview(
        self,
        capture: PreviewCapture,
    ) -> bool:
        tab = capture.tab_id
        if self._close_fenced:
            return False
        pending_interactive = self._pending_interactive_preview
        if pending_interactive is not None and pending_interactive.tab_id == tab:
            self._pending_interactive_preview = None
        self._pending_preview = capture
        self._mark_unsettled()
        # An active Run completes and becomes the Reference; the Preview
        # waits in its pending slot and is redriven after the Run settles.
        self._retire_if_affected(
            tab,
            (target.set_id for target in capture.targets),
            include_run=False,
        )
        self._retire_unserved_reference_run()
        if self._worker is not None or not self._gate.is_idle:
            self.status_changed.emit("Preview pending: latest exact request retained.")
            self.state_changed.emit()
            return False
        return self._redrive_pending_preview()

    def _redrive_pending_preview(self) -> bool:
        capture = self._pending_preview
        if capture is None or self._worker is not None or not self._gate.is_idle or self._close_fenced:
            return False
        tab_id = capture.tab_id
        targets = capture.targets
        repository = self._repository(tab_id)
        preview_ids = set(repository.preview_set_ids())
        missing_targets: list[PreviewTarget] = []
        unavailable: list[str] = []
        for target in targets:
            canonical = target.canonical_run_target.execution_request
            reference_request = repository.reference_request(target.set_id)
            if reference_request is not None and (reference_request == canonical):
                continue
            current_request = repository.current_request(target.set_id)
            if (
                target.set_id not in preview_ids
                and current_request is not None
                and (current_request == canonical)
            ):
                continue
            failed = self._failed_run_requests.get((tab_id, target.set_id))
            if failed is not None and (failed == canonical):
                unavailable.append(target.set_name)
                continue
            missing_targets.append(target)
        if missing_targets:
            started = self._start(
                tab_id,
                "run",
                tuple(target.canonical_run_target for target in missing_targets),
                max_parallel=self._capacity(len(missing_targets)),
                purpose="preview-reference",
            )
            if not started:
                self._pending_preview = None
                self.reconcile_canonical_targets(
                    tab_id, tuple(target.set_id for target in targets),
                )
            return started

        self._pending_preview = None
        started = self._start(
            tab_id,
            "preview",
            targets,
            max_parallel=1,
        )
        if not started:
            self._pending_preview = capture
        else:
            self._operation["details"].extend(
                f"Reference unavailable for {name}" for name in unavailable
            )
        return started

    def request_interactive_preview(
        self,
        request: InteractivePreviewRequest,
    ) -> bool:
        """Run one admitted transient Slider Preview through this owner."""

        if self._close_fenced:
            return False
        if (
            self._pending_preview is not None
            and self._pending_preview.tab_id == request.tab_id
        ):
            self._pending_preview = None
        self._pending_interactive_preview = request
        self._mark_unsettled()
        # A user-started Run is never cancelled by a Slider Preview; a
        # Reference Run dies with the exact Preview this request discarded.
        self._retire_if_affected(
            request.tab_id,
            request.target_set_ids,
            include_run=False,
        )
        self._retire_unserved_reference_run()
        if self._worker is not None or not self._gate.is_idle:
            self.status_changed.emit(
                "Interactive Preview pending: latest request retained."
            )
            self.state_changed.emit()
            return False
        return self._redrive_pending_interactive_preview()

    def _redrive_pending_interactive_preview(self) -> bool:
        request = self._pending_interactive_preview
        if (
            request is None
            or self._worker is not None
            or not self._gate.is_idle
            or self._close_fenced
        ):
            return False
        self._pending_interactive_preview = None
        started = self._start(
            request.tab_id,
            "preview",
            request.targets,
            max_parallel=self._capacity(len(request.targets)),
            purpose="interactive-preview",
            payload=request,
        )
        if not started:
            self._pending_interactive_preview = request
        return started

    def retire_interactive_preview(
        self,
        tab_id: str | None = None,
        set_ids: Iterable[str] | None = None,
    ) -> bool:
        """Retire transient Slider Preview computation and display targets."""

        tab = tab_id
        ids = None if set_ids is None else set(set_ids)
        changed = False
        operation = self._operation
        if (
            operation is not None
            and operation["purpose"] == "interactive-preview"
            and (tab is None or operation["tab_id"] == tab)
        ):
            changed = self._retire_if_affected(
                operation["tab_id"],
                (
                    (target.set_id for target in operation["targets"])
                    if ids is None else ids
                ),
            )
        pending = self._pending_interactive_preview
        if (
            pending is not None
            and (tab is None or pending.tab_id == tab)
        ):
            pending_ids = tuple(target.set_id for target in pending.targets)
            affected = pending_ids if ids is None else tuple(
                set_id for set_id in pending_ids if set_id in ids
            )
            if affected:
                self._pending_interactive_preview = None
                changed = True
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def retire_preview(
        self,
        tab_id: str,
        set_ids: Iterable[str] | None = None,
    ) -> bool:
        """Retire affected exact Preview work and reconcile its display truth."""

        tab = tab_id
        ids = None if set_ids is None else tuple(dict.fromkeys(set_ids))
        interactive_changed = self.retire_interactive_preview(tab, ids)
        changed = interactive_changed
        restore_ids: tuple[str, ...] = ()
        operation = self._operation
        if (
            operation is not None
            and str(operation["kind"]) == "preview"
            and str(operation["purpose"]) != "interactive-preview"
        ):
            operation_ids = tuple(
                target.set_id for target in operation["targets"]
            )
            affected = operation_ids if ids is None else ids
            restore_ids = tuple(value for value in affected if value in operation_ids)
            changed = (
                self._retire_if_affected(tab, restore_ids, include_run=False)
                or changed
            )
        pending = self._pending_preview
        if pending is not None and pending.tab_id == tab:
            pending_ids = tuple(target.set_id for target in pending.targets)
            affected = pending_ids if ids is None else ids
            pending_restore_ids = tuple(value for value in affected if value in pending_ids)
            if pending_restore_ids:
                self._pending_preview = None
                changed = True
                restore_ids = tuple(dict.fromkeys((*restore_ids, *pending_restore_ids)))
        repository = self._repository(tab)
        preview_ids = repository.preview_set_ids()
        if ids is None:
            preview_restore_ids = preview_ids
        else:
            preview_restore_ids = tuple(
                set_id for set_id in preview_ids if set_id in set(ids)
            )
        restore_ids = tuple(
            dict.fromkeys((*restore_ids, *preview_restore_ids))
        )
        if restore_ids:
            self.reconcile_canonical_targets(tab, restore_ids)
            changed = True
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def cancel_current(self) -> bool:
        changed = bool(
            self._worker is not None
            or self._pending_preview is not None
            or self._pending_interactive_preview is not None
        )
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
        self._pending_preview = None
        self._pending_interactive_preview = None
        self._terminal_state = "cancelled"
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def cancel_for_application_close(self) -> None:
        self._close_fenced = True
        self.cancel_current()

    def show_cached(self, tab_id: str, set_id: str, execution_request: SimulationExecutionRequest) -> bool:
        result = self._repository(tab_id).hydrate_cached(set_id, execution_request)
        if result is None:
            # The caller still renders the set from its retained result.
            return False
        self.results_changed.emit(str(tab_id))
        return True

    @QtCore.Slot(int, object)
    def _on_target_terminal(self, generation: int, outcome: _TargetTerminalOutcome) -> None:
        if not self._operation_is_current(generation):
            return
        operation = self._operation
        target = outcome.target
        interactive = operation["purpose"] == "interactive-preview"
        request = operation["payload"]
        # A newer request may remain pending while this worker drains. Its
        # terminal outcomes are no longer presentation authority, even when
        # its target roster does not overlap the older operation.
        if (
            interactive
            and self._pending_interactive_preview is not None
            and self._pending_interactive_preview is not request
        ):
            operation["outcomes"].append((target, "discarded"))
            return
        if outcome.state != "success":
            state = outcome.state
            if state == "failure":
                if interactive:
                    self.interactive_target_failed.emit(
                        operation["tab_id"], target.set_id, request,
                    )
                elif operation["kind"] == "run":
                    self._failed_run_requests[
                        (operation["tab_id"], target.set_id)
                    ] = target.execution_request
            operation["outcomes"].append((target, state))
            if outcome.message:
                operation["details"].append(f"{target.set_name}: {outcome.message}")
            self._terminal_state = state
            return
        termination = outcome.result.termination
        if termination.cause == "stop":
            operation["details"].append(
                f"{target.set_name}: completed at t={termination.actual_endpoint:g} s "
                "(authored stopping intervention)"
            )
        if interactive:
            self.interactive_result_ready.emit(
                operation["tab_id"],
                target.set_id,
                request,
                outcome.result,
            )
            operation["outcomes"].append((target, "success"))
            self._terminal_state = "success"
            return
        if operation["kind"] == "preview":
            operation["preview_results"][target.set_id] = outcome.result
        else:
            repository = self._repository(operation["tab_id"])
            repository.publish_run(
                target.set_id, target.execution_request, outcome.result
            )
            # A success of the remembered failed request supersedes that failure.
            key = (operation["tab_id"], target.set_id)
            failed = self._failed_run_requests.get(key)
            if failed is not None and (failed == target.execution_request):
                del self._failed_run_requests[key]
        operation["outcomes"].append((target, "success"))
        self._terminal_state = "success"
        if operation["kind"] == "run":
            self.results_changed.emit(operation["tab_id"])

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        if self.sender() is not self._worker:
            return
        operation = self._operation
        operation_kind = operation["kind"]
        operation_purpose = operation["purpose"]
        operation_tab_id = operation["tab_id"]
        removed_ids = operation["removed_target_ids"]
        seen = {id(target) for target, _state in operation["outcomes"]}
        for target in operation["targets"]:
            if id(target) not in seen:
                operation["outcomes"].append((target, "cancelled" if self._terminal_state == "cancelled" else "failure"))
        superseded_preview = any(
            pending is not None and pending.tab_id == operation_tab_id
            for pending in (
                self._pending_preview,
                self._pending_interactive_preview,
            )
        )
        ordinary_preview = operation_kind == "preview" and operation_purpose == "ordinary"
        preview_current = ordinary_preview and self._operation_is_current(
            operation["generation"]
        )
        if ordinary_preview and not preview_current:
            operation["outcomes"] = [
                (
                    target,
                    "discarded" if outcome_state == "success" else outcome_state,
                )
                for target, outcome_state in operation["outcomes"]
            ]
        states = [state for _target, state in operation["outcomes"]]
        succeeded = states.count("success")
        failed = states.count("failure")
        cancelled = states.count("cancelled")
        discarded = states.count("discarded")
        total = succeeded + failed + cancelled + discarded
        state = "success" if total and succeeded == total else "failure" if total and failed == total else "cancelled" if total and cancelled == total else "partial"
        summary = SimulationTerminalSummary(operation_kind, state, succeeded, failed, cancelled, discarded, tuple(operation["details"]))
        preview_published = False
        if preview_current and summary.state == "success" and not superseded_preview:
            repository = self._repository(operation_tab_id)
            preview_results = operation["preview_results"]
            for target in operation["targets"]:
                repository.retain_explicit_as_reference(target.set_id)
                repository.publish_preview(
                    target.set_id,
                    target.execution_request,
                    preview_results[target.set_id],
                )
            preview_published = True
        reconcile_target_ids: tuple[str, ...] = ()
        if ordinary_preview and summary.state != "success":
            reconcile_target_ids = tuple(dict.fromkeys(
                target.set_id
                for target in operation["targets"]
                if target.set_id not in removed_ids
            ))
        worker = self._worker
        self._worker = None
        self._operation = None
        self._terminal_state = None
        self._gate.release(self)
        worker.deleteLater()
        self.state_changed.emit()
        self._emit_settled_if_complete()
        if not self._close_fenced:
            if operation_purpose == "preview-reference":
                self.status_changed.emit(
                    "Reference " + self._terminal_status(summary)
                )
                # Continue synchronously after releasing the worker and gate;
                # queued redrives then see the new worker or an empty request.
                self._redrive_pending_preview()
                if self._pending_interactive_preview is not None:
                    QtCore.QTimer.singleShot(
                        0, self._redrive_pending_interactive_preview,
                    )
            elif operation_purpose == "interactive-preview":
                self.status_changed.emit(self._terminal_status(summary))
                if self._pending_interactive_preview is not None:
                    QtCore.QTimer.singleShot(
                        0,
                        self._redrive_pending_interactive_preview,
                    )
                elif self._pending_preview is not None:
                    QtCore.QTimer.singleShot(0, self._redrive_pending_preview)
            else:
                if preview_published:
                    self.results_changed.emit(operation_tab_id)
                elif reconcile_target_ids:
                    self.reconcile_canonical_targets(
                        operation_tab_id,
                        reconcile_target_ids,
                        clear_pending=False,
                    )
                self.status_changed.emit(self._terminal_status(summary))
                if self._pending_preview is not None and operation_kind == "run":
                    # The waiting Preview uses this Run's outcomes as its
                    # Reference; failed requests are remembered, not retried.
                    self._redrive_pending_preview()
                elif self._pending_preview is not None:
                    QtCore.QTimer.singleShot(0, self._redrive_pending_preview)
                elif self._pending_interactive_preview is not None:
                    QtCore.QTimer.singleShot(
                        0,
                        self._redrive_pending_interactive_preview,
                    )

    @staticmethod
    def _terminal_status(summary: SimulationTerminalSummary) -> str:
        state = "canceled" if summary.state == "cancelled" else summary.state
        status = f"{summary.kind.capitalize()} {state}: {summary.succeeded} succeeded, {summary.failed} failed, {summary.cancelled} canceled, {summary.discarded} discarded."
        if summary.details:
            status += " " + " ".join(item if item.endswith(".") else item + "." for item in summary.details)
        return status

    @QtCore.Slot()
    def _on_gate_changed(self) -> None:
        if self._gate.is_idle and self._worker is None:
            if self._pending_preview is not None:
                QtCore.QTimer.singleShot(0, self._redrive_pending_preview)
            elif self._pending_interactive_preview is not None:
                QtCore.QTimer.singleShot(
                    0,
                    self._redrive_pending_interactive_preview,
                )
