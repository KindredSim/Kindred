"""Direct Run and exact Preview lifecycle owner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from threading import Event
from typing import Callable, Iterable, Literal

from PySide6 import QtCore

from kindred.core.exceptions import SimulationCancelled
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
)
from kindred.core.project_session import ProjectContentToken
from kindred.core.result_repository import ResultRepository, SimulationResultValue
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.simulation_algebra_policy import SimulationAlgebraPolicy
from kindred.core.simulation_failure import FactualCause, factual_cause_from_exception
from kindred.core.simulation_preparation import (
    PreparedSimulationRun,
    SimulationExecutionRequest,
    SimulationPreparationError,
    build_simulation_request_from_prepared_run,
    build_simulation_execution_request,
    prepare_simulation_worker_run,
    resolve_prepared_run_intervention_execution,
    simulation_execution_requests_equal,
)
from kindred.core.simulation_result_finalization import finalize_simulation_result
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.interactive_preview import (
    InteractivePreviewRequest,
    InteractivePreviewTarget,
)
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.workspace_tab_registry import ScientificDocumentDelta

__all__ = [
    "ExplicitRunCapture",
    "PreviewCapture",
    "SimulationJobOwner",
    "build_explicit_target",
]


def _required_id(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be nonblank.")
    return result


def resolve_run_parallel_capacity(*, cpu_count: object, user_max: object, target_count: int) -> int:
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("Run requires a positive exact target count.")
    cpu = int(cpu_count) if type(cpu_count) is int and cpu_count > 0 else 1
    budget = max(1, cpu - 1)
    default = min(budget, 16)
    resolved = max(1, min(int(user_max), MAX_PARALLEL_WORKERS_CEILING)) if type(user_max) is int else default
    return min(target_count, budget, resolved, MAX_PARALLEL_WORKERS_CEILING)


@dataclass(frozen=True, slots=True)
class ExplicitRunTarget:
    tab_id: str
    set_id: str
    set_name: str
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(self, "set_name", _required_id(self.set_name, "set_name"))
        request = self.execution_request
        if request.parameter_overrides:
            raise ValueError("explicit Run targets cannot carry Preview values.")
        object.__setattr__(self, "execution_request", request)


@dataclass(frozen=True, slots=True)
class ExplicitRunCapture:
    tab_id: str
    content_token: ProjectContentToken
    targets: tuple[ExplicitRunTarget, ...]

    def __post_init__(self) -> None:
        tab = _required_id(self.tab_id, "tab_id")
        if not isinstance(self.content_token, ProjectContentToken):
            raise TypeError("content_token must be ProjectContentToken.")
        targets = tuple(self.targets)
        if not targets or any(not isinstance(target, ExplicitRunTarget) for target in targets):
            raise TypeError("Run requires ExplicitRunTarget values.")
        if any(target.tab_id != tab for target in targets):
            raise ValueError("every Run target must belong to the capture tab.")
        ids = tuple(target.set_id for target in targets)
        if len(ids) != len(set(ids)):
            raise ValueError("Run target set IDs must be unique.")
        object.__setattr__(self, "tab_id", tab)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class PreviewTarget:
    tab_id: str
    set_id: str
    set_name: str
    execution_request: SimulationExecutionRequest | None
    canonical_run_target: ExplicitRunTarget

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(self, "set_name", _required_id(self.set_name, "set_name"))


@dataclass(frozen=True, slots=True)
class PreviewCapture:
    """One exact Preview action, before or after B request admission."""

    tab_id: str
    content_token: ProjectContentToken
    targets: tuple[PreviewTarget, ...]

    def __post_init__(self) -> None:
        tab = _required_id(self.tab_id, "tab_id")
        if not isinstance(self.content_token, ProjectContentToken):
            raise TypeError("content_token must be ProjectContentToken.")
        targets = tuple(self.targets)
        if not targets or any(not isinstance(item, PreviewTarget) for item in targets):
            raise TypeError("Preview requires PreviewTarget values.")
        if any(item.tab_id != tab for item in targets):
            raise ValueError("every Preview target must belong to the capture tab.")
        ids = tuple(item.set_id for item in targets)
        if len(ids) != len(set(ids)):
            raise ValueError("Preview target set IDs must be unique.")
        object.__setattr__(self, "tab_id", tab)
        object.__setattr__(self, "targets", targets)


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
    stage: Literal["preparation", "execution"]
    result: SimulationResultValue | None = None
    message: str = ""
    cause: FactualCause | None = None


def _simulation_worker_cause(exc: BaseException) -> FactualCause:
    cause = factual_cause_from_exception(exc, domain="simulation", origin="simulation_worker")
    if not isinstance(exc, SimulationPreparationError):
        return cause
    details = dict(cause.details)
    details.update({"stage": str(exc.stage or "unknown"), "parameter_names": tuple(sorted(exc.parameter_names))})
    return FactualCause(
        domain=cause.domain, origin=cause.origin, code=cause.code, type=cause.type,
        message=cause.message, context=cause.context, details=details, cause=cause.cause,
    )


def build_explicit_target(*, prepared_dsl: PreparedDslRequestSet, tab_id: str, set_id: str, set_name: str,
                          mechanism_source: MechanismAuthoringSource, initial_values: Iterable[tuple[str, float]],
                          execution_settings: SimulationExecutionSettings, scientific_settings: TabScientificSettings) -> ExplicitRunTarget:
    execution_request = build_simulation_execution_request(
        prepared_dsl=prepared_dsl, mechanism_source=mechanism_source,
        initial_values=initial_values,
        execution_settings=execution_settings, scientific_settings=scientific_settings,
    )
    return ExplicitRunTarget(tab_id, set_id, set_name, execution_request)


def _captured_parameter_coordinates(request: SimulationExecutionRequest) -> tuple[tuple[str, float, str], ...]:
    coordinates = enumerate_parameter_coordinates(request.prepared_dsl)
    by_name = {coordinate.name: coordinate for coordinate in coordinates}
    overrides = {str(name): float(value) for name, value in dict(request.parameter_overrides or {}).items()}
    for name in overrides:
        coordinate = by_name.get(name)
        if coordinate is None:
            raise ValueError(f"Preview parameter {name!r} is not a direct GUI coordinate.")
        if not coordinate.writable or coordinate.derived:
            raise ValueError(
                f"Preview parameter {name!r} is owned by retained Algebra."
            )
    return tuple((coordinate.name, overrides.get(coordinate.name, float(coordinate.value)), coordinate.unit) for coordinate in coordinates)


def _prepare_target(
    target: ExplicitRunTarget | PreviewTarget | InteractivePreviewTarget,
    *,
    cancellation_control: SimulationCancellationControl,
) -> tuple[PreparedSimulationRun, tuple[tuple[str, float, str], ...]]:
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    request = target.execution_request
    parameters = _captured_parameter_coordinates(request)
    prepared = prepare_simulation_worker_run(
        execution_request=request,
        cancellation_control=cancellation_control,
    )
    return prepared, parameters


def _solve_prepared(prepared: PreparedSimulationRun, *, parameters: tuple[tuple[str, float, str], ...], cancellation_control: SimulationCancellationControl) -> SimulationResultValue:
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    intervention_execution = prepared.intervention_execution
    if intervention_execution is None:
        intervention_execution = resolve_prepared_run_intervention_execution(
            prepared,
            parameter_values={name: value for name, value, _unit in parameters},
            cancellation_check=cancellation_control.cancel_requested,
        )
    solved = solve_ode(
        build_simulation_request_from_prepared_run(
            prepared,
            y0=prepared.y0,
            intervention_execution=intervention_execution,
            symbolic_jacobian=prepared.symbolic_jacobian,
            cancellation_control=cancellation_control,
        )
    )
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    result = finalize_simulation_result(
        mechanism=prepared.mechanism, result=solved, species_names=list(prepared.species_names),
        initials_for_algebra=prepared.initials_for_algebra, temperature_schedule=prepared.temperature_schedule,
        parameter_algebra_spec=prepared.parameter_algebra_spec, algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        parameters=parameters, preparation_warnings=prepared.warnings,
    )
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    return result


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
        cancellation_control: SimulationCancellationControl,
        cancel_event: Event,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._generation = int(generation)
        self._targets = tuple(targets)
        self._max_parallel = max(1, min(int(max_parallel), len(self._targets)))
        self._control = cancellation_control
        self._cancel_event = cancel_event

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
                        stage="preparation",
                        message="Simulation cancelled.",
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
                        stage="preparation", message=str(exc) or type(exc).__name__,
                        cause=None if self.cancelled else _simulation_worker_cause(exc),
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
            return _TargetTerminalOutcome(target, "cancelled", "preparation", message="Simulation cancelled.")
        try:
            prepared, parameters = _prepare_target(target, cancellation_control=self._control)
        except SimulationCancelled:
            return _TargetTerminalOutcome(target, "cancelled", "preparation", message="Simulation cancelled.")
        except BaseException as exc:  # noqa: BLE001
            return _TargetTerminalOutcome(target, "failure", "preparation", message=str(exc) or type(exc).__name__, cause=_simulation_worker_cause(exc))
        try:
            result = _solve_prepared(prepared, parameters=parameters, cancellation_control=self._control)
        except SimulationCancelled:
            return _TargetTerminalOutcome(target, "cancelled", "execution", message="Simulation cancelled.")
        except BaseException as exc:  # noqa: BLE001
            return _TargetTerminalOutcome(target, "failure", "execution", message=str(exc) or type(exc).__name__, cause=_simulation_worker_cause(exc))
        return _TargetTerminalOutcome(target, "success", "execution", result=result)


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

    def __init__(self, *, gate: ComputationGate, repository_for_tab: RepositoryResolver,
                 user_parallel_maximum: RawUserParallelMaximum | None = None,
                 cpu_count: Callable[[], object] | None = None,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        if not isinstance(gate, ComputationGate):
            raise TypeError("gate must be a ComputationGate.")
        if not callable(repository_for_tab):
            raise TypeError("repository_for_tab must be callable.")
        self._gate = gate
        self._repository_for_tab = repository_for_tab
        self._user_parallel_maximum = user_parallel_maximum or (lambda: None)
        self._cpu_count = cpu_count or os.cpu_count
        self._generation = 0
        self._operation: dict[str, object] | None = None
        self._worker: _DirectSimulationWorker | None = None
        self._cancel_event: Event | None = None
        self._cancellation_control: SimulationCancellationControl | None = None
        self._terminal_state: str | None = None
        self._pending_preview: PreviewCapture | None = None
        self._pending_interactive_preview: InteractivePreviewRequest | None = None
        self._close_fenced = False
        self._last_close_settled = True
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

    def _mark_unsettled(self) -> None:
        self._last_close_settled = False

    def _emit_settled_if_complete(self) -> None:
        if not self._last_close_settled and self.application_close_settled:
            self._last_close_settled = True
            self.application_close_settlement_changed.emit()

    def _repository(self, tab_id: str) -> ResultRepository:
        repository = self._repository_for_tab(str(tab_id))
        if not isinstance(repository, ResultRepository):
            raise LookupError(f"No ResultRepository exists for tab {tab_id!r}.")
        return repository

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
        tab = _required_id(tab_id, "tab_id")
        ids = {_required_id(value, "set_id") for value in set_ids}
        if not ids:
            return False
        operation = self._operation
        if operation is None or str(operation["tab_id"]) != tab:
            return False
        if not include_run and str(operation["kind"]) != "preview":
            return False
        if not ids.intersection(target.set_id for target in operation["targets"]):
            return False
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
        self._terminal_state = "cancelled"
        return True

    def _retire_pending_preview_if_affected(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> bool:
        pending = self._pending_preview
        if pending is None or pending.tab_id != str(tab_id):
            return False
        ids = {
            _required_id(value, "set_id")
            for value in set_ids
        }
        if not ids.intersection(target.set_id for target in pending.targets):
            return False
        self._pending_preview = None
        return True

    def fence_document_delta(self, delta: ScientificDocumentDelta) -> bool:
        """Fence computation for one admitted document delta.

        The registry owns repository mutation and physical presentation
        retirement.  This operation only retires computation intent and
        records removed identities on the operation being cancelled.
        """

        targets_by_tab: dict[str, list[str]] = {}
        removed_by_tab: dict[str, set[str]] = {}
        for entries, removed in (
            (delta.affected_targets_by_tab, False),
            (delta.removed_targets_by_tab, True),
        ):
            for tab_id, set_ids in entries:
                tab = str(tab_id)
                targets = targets_by_tab.setdefault(tab, [])
                removed_ids = removed_by_tab.setdefault(tab, set())
                for set_id in set_ids:
                    value = str(set_id)
                    if value not in targets:
                        targets.append(value)
                    if removed:
                        removed_ids.add(value)

        changed = False
        state_changed = False
        operation = self._operation
        if operation is not None:
            operation_tab = str(operation["tab_id"])
            recorded_removed = operation.setdefault(
                "removed_target_ids", set()
            )
            new_removed = set(removed_by_tab.get(operation_tab, ())) - recorded_removed
            if new_removed:
                recorded_removed.update(new_removed)
                changed = True
            affected = tuple(targets_by_tab.get(operation_tab, ()))
            operation_ids = {
                target.set_id for target in operation["targets"]
            }
            operation_affected = tuple(
                set_id for set_id in affected if set_id in operation_ids
            )
            if operation_affected:
                if (
                    str(operation["kind"]) == "preview"
                    and str(operation["purpose"]) == "interactive-preview"
                ):
                    retired_ids = operation.setdefault("retired_ids", set())
                    new_retired = set(operation_affected) - retired_ids
                    if new_retired:
                        retired_ids.update(new_retired)
                        changed = True

                if not operation.get("document_delta_cancelled", False):
                    self._retire_if_affected(
                        operation_tab,
                        operation_affected,
                    )
                    operation["document_delta_cancelled"] = True
                    changed = True
                    state_changed = True

        for tab, target_ids in targets_by_tab.items():
            affected_ids = set(target_ids)
            pending_preview = self._pending_preview
            if (
                pending_preview is not None
                and pending_preview.tab_id == tab
                and affected_ids.intersection(
                    target.set_id for target in pending_preview.targets
                )
            ):
                self._pending_preview = None
                changed = True
                state_changed = True
            pending_interactive = self._pending_interactive_preview
            if (
                pending_interactive is not None
                and pending_interactive.tab_id == tab
                and affected_ids.intersection(pending_interactive.target_set_ids)
            ):
                self._pending_interactive_preview = None
                changed = True
                state_changed = True

        if state_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def reconcile_canonical_targets(self, tab_id: str, set_ids: Iterable[str], *, clear_pending: bool = True) -> tuple[tuple[str, bool], ...]:
        ids = tuple(dict.fromkeys(_required_id(value, "set_id") for value in set_ids))
        operation = self._operation
        if not (
            operation is not None
            and str(operation["kind"]) == "preview"
            and str(operation["purpose"]) == "interactive-preview"
        ):
            self._retire_if_affected(tab_id, ids, include_run=False)
        interactive_changed = self.retire_interactive_preview(tab_id, ids)
        pending_changed = clear_pending and self._retire_pending_preview_if_affected(tab_id, ids)
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
            cpu_count=self._cpu_count(),
            user_max=self._user_parallel_maximum(),
            target_count=count,
        )

    def _promote_preview_targets(
        self,
        tab_id: str,
        targets: tuple[PreviewTarget, ...],
    ) -> None:
        """Establish exact Preview semantic authority at worker start."""

        repository = self._repository(tab_id)
        set_ids = tuple(target.set_id for target in targets)
        for set_id in set_ids:
            repository.retain_explicit_as_reference(set_id)
        repository.deauthorize_sets(set_ids, preserve_reference=True)
        self.results_changed.emit(str(tab_id))

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
        event = Event()
        control = SimulationCancellationControl(event.is_set)
        worker = _DirectSimulationWorker(generation=generation, targets=targets, max_parallel=max_parallel, cancellation_control=control, cancel_event=event, parent=self)
        self._operation = {
            "generation": generation,
            "kind": kind,
            "purpose": purpose,
            "tab_id": str(tab_id),
            "targets": targets,
            "outcomes": [],
            "details": [],
            "payload": payload,
            "removed_target_ids": set(),
            "retired_ids": set(),
            "document_delta_cancelled": False,
        }
        self._worker = worker
        self._cancel_event = event
        self._cancellation_control = control
        self._terminal_state = None
        worker.target_terminal.connect(self._on_target_terminal)
        worker.finished.connect(self._on_worker_finished)
        try:
            worker.start()
        except BaseException as exc:  # noqa: BLE001
            if self._worker is not None:
                self._generation += 1
                self._worker.cancel()
                self._terminal_state = "failure"
            self._worker = None
            self._operation = None
            self._cancel_event = None
            self._cancellation_control = None
            self._gate.release(self)
            self.status_changed.emit(f"{kind.capitalize()} dispatch failed: {exc}")
            self._emit_settled_if_complete()
            return False
        if kind == "run" and purpose == "ordinary":
            repository = self._repository(str(tab_id))
            repository.deauthorize_sets(
                tuple(target.set_id for target in targets),
            )
            self.results_changed.emit(str(tab_id))
        elif kind == "preview" and purpose != "interactive-preview":
            self._promote_preview_targets(
                str(tab_id),
                tuple(targets),
            )
        self.state_changed.emit()
        if kind == "run":
            self.run_dispatched.emit(str(tab_id), tuple(target.set_id for target in targets))
        return True

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
        operation = self._operation
        if operation is None or operation["purpose"] != "preview-prerequisite":
            self._retire_if_affected(
                tab,
                (target.set_id for target in capture.targets),
            )
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
        set_ids = tuple(target.set_id for target in targets)

        repository = self._repository(tab_id)
        preview_ids = set(repository.preview_set_ids())
        matching: set[str] = set()
        for target in targets:
            reference_request = repository.reference_request(target.set_id)
            reference_result = repository.reference_result(target.set_id)
            canonical_target = target.canonical_run_target
            if (
                reference_request is not None
                and reference_result is not None
                and simulation_execution_requests_equal(
                    reference_request,
                    canonical_target.execution_request,
                )
            ):
                matching.add(target.set_id)
                continue
            if target.set_id in preview_ids:
                continue
            current_request = repository.current_request(target.set_id)
            current_result = repository.current_result(target.set_id)
            if (
                current_request is not None
                and current_result is not None
                and simulation_execution_requests_equal(
                    current_request,
                    canonical_target.execution_request,
                )
            ):
                matching.add(target.set_id)
        missing_targets = tuple(
            target
            for target in targets
            if target.set_id not in matching
        )
        if missing_targets:
            started = self._start(
                tab_id,
                "run",
                tuple(target.canonical_run_target for target in missing_targets),
                max_parallel=self._capacity(len(missing_targets)),
                purpose="preview-prerequisite",
            )
            if not started:
                self._pending_preview = None
                self.reconcile_canonical_targets(tab_id, set_ids)
                return False
            return True

        self._pending_preview = None
        started = self._start(
            tab_id,
            "preview",
            targets,
            max_parallel=1,
        )
        if not started:
            self._pending_preview = capture
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
        self._retire_if_affected(
            request.tab_id,
            request.target_set_ids,
        )
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

        tab = None if tab_id is None else _required_id(tab_id, "tab_id")
        ids = None if set_ids is None else {
            _required_id(value, "set_id") for value in set_ids
        }
        changed = False
        operation = self._operation
        if (
            operation is not None
            and str(operation["kind"]) == "preview"
            and str(operation["purpose"]) == "interactive-preview"
            and (tab is None or str(operation["tab_id"]) == tab)
        ):
            operation_ids = tuple(target.set_id for target in operation["targets"])
            affected = operation_ids if ids is None else tuple(
                set_id for set_id in operation_ids if set_id in ids
            )
            retired_ids = set(operation.get("retired_ids", set()))
            affected = tuple(
                set_id for set_id in affected if set_id not in retired_ids
            )
            if affected:
                changed = self._retire_if_affected(
                    str(operation["tab_id"]),
                    affected,
                )
                if changed:
                    operation.setdefault("retired_ids", set()).update(affected)
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
        *,
        preserve_reference: bool = False,
    ) -> bool:
        """Retire affected exact Preview work and reconcile its display truth."""

        tab = _required_id(tab_id, "tab_id")
        ids = None if set_ids is None else tuple(
            dict.fromkeys(_required_id(value, "set_id") for value in set_ids)
        )
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
            if preserve_reference and changed:
                operation["purpose"] = "interactive-successor"
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
        if preserve_reference:
            for set_id in restore_ids:
                repository.clear_preview(set_id, clear_reference=False)
            changed = bool(restore_ids) or changed
            if restore_ids:
                self.results_changed.emit(tab)
        elif restore_ids:
            self.reconcile_canonical_targets(tab, restore_ids)
            changed = True
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def cancel_current(self) -> bool:
        operation = self._operation
        changed = bool(
            self._worker is not None
            or self._pending_preview is not None
            or self._pending_interactive_preview is not None
        )
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
        if (
            operation is not None
            and str(operation["kind"]) == "preview"
            and str(operation["purpose"]) == "interactive-preview"
        ):
            operation.setdefault("retired_ids", set()).update(
                target.set_id for target in operation["targets"]
            )
        self._pending_preview = None
        self._pending_interactive_preview = None
        self._terminal_state = "cancelled"
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def fence_application_close_admission(self) -> None:
        self._close_fenced = True
        self.cancel_current()

    def cancel_for_application_close(self) -> None:
        self.fence_application_close_admission()

    def show_cached(self, tab_id: str, set_id: str, execution_request: SimulationExecutionRequest) -> bool:
        result = self._repository(tab_id).hydrate_cached(set_id, execution_request)
        if result is None:
            self.status_changed.emit("Requested result is not cached; display unchanged.")
            return False
        self.results_changed.emit(str(tab_id))
        return True

    @QtCore.Slot(int, object)
    def _on_target_terminal(self, generation: int, outcome: object) -> None:
        if not isinstance(outcome, _TargetTerminalOutcome) or not self._operation_is_current(generation):
            return
        operation = self._operation
        if operation is None:
            return
        target = outcome.target
        if outcome.state != "success" or not isinstance(outcome.result, SimulationResultValue):
            state = "cancelled" if outcome.state == "cancelled" else "failure"
            operation["outcomes"].append((target, state))
            if outcome.message:
                operation["details"].append(f"{target.set_name}: {outcome.message}")
            self._terminal_state = state
            return
        # The worker operation sequence fences late terminal events immediately
        # before direct repository publication; action currentness is carried by
        # the admitted capture consumed by its initiating owner.
        if self._close_fenced:
            operation["outcomes"].append((target, "discarded"))
            return
        if (
            operation["kind"] == "preview"
            and operation["purpose"] == "interactive-preview"
        ):
            request = operation["payload"]
            self.interactive_result_ready.emit(
                str(operation["tab_id"]),
                target.set_id,
                request,
                outcome.result,
            )
            operation["outcomes"].append((target, "success"))
            self._terminal_state = "success"
            return
        try:
            repository = self._repository(str(operation["tab_id"]))
            if operation["kind"] == "run":
                repository.publish_run(target.set_id, target.execution_request, outcome.result)
            else:
                repository.publish_preview(target.set_id, target.execution_request, outcome.result)
        except Exception as exc:  # noqa: BLE001
            operation["outcomes"].append((target, "discarded"))
            operation["details"].append(f"{target.set_name}: repository rejected publication ({exc})")
            return
        operation["outcomes"].append((target, "success"))
        self._terminal_state = "success"
        self.results_changed.emit(str(operation["tab_id"]))

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        if self.sender() is not self._worker:
            return
        operation = self._operation
        prerequisite_failed = False
        operation_kind = "run"
        operation_purpose: str | None = None
        operation_tab_id: str | None = None
        reconcile_target_ids: tuple[str, ...] = ()
        interactive_successor_ids: tuple[str, ...] = ()
        if operation is not None:
            operation_kind = str(operation["kind"])
            operation_purpose = str(operation["purpose"])
            operation_tab_id = str(operation["tab_id"])
            removed_ids = set(operation.get("removed_target_ids", set()))
            seen = {id(target) for target, _state in operation["outcomes"]}
            for target in operation["targets"]:
                if id(target) not in seen:
                    operation["outcomes"].append((target, "cancelled" if self._terminal_state == "cancelled" else "failure"))
            states = [state for _target, state in operation["outcomes"]]
            succeeded = states.count("success")
            failed = states.count("failure")
            cancelled = states.count("cancelled")
            discarded = states.count("discarded")
            total = succeeded + failed + cancelled + discarded
            state = "success" if total and succeeded == total else "failure" if total and failed == total else "cancelled" if total and cancelled == total else "partial"
            summary = SimulationTerminalSummary(operation_kind, state, succeeded, failed, cancelled, discarded, tuple(operation["details"]))
            target_ids = tuple(
                target.set_id
                for target in operation["targets"]
                if target.set_id not in removed_ids
            )
            target_ids = tuple(dict.fromkeys(target_ids))
            if operation_purpose == "preview-prerequisite":
                prerequisite_failed = summary.state != "success"
                if prerequisite_failed:
                    pending = self._pending_preview
                    self._pending_preview = None
                    if pending is not None and pending.tab_id == str(operation["tab_id"]):
                        target_ids = tuple(
                            dict.fromkeys(
                                (
                                    *target_ids,
                                    *(
                                        target.set_id
                                        for target in pending.targets
                                        if target.set_id not in removed_ids
                                    ),
                                )
                            )
                        )
                    reconcile_target_ids = target_ids

            elif (
                operation_kind == "preview"
                and operation_purpose != "interactive-preview"
                and summary.state != "success"
            ):
                if operation_purpose == "interactive-successor":
                    interactive_successor_ids = target_ids
                else:
                    reconcile_target_ids = target_ids
        else:
            summary = SimulationTerminalSummary("run", "cancelled", 0, 0, 1, 0)
        worker = self._worker
        self._worker = None
        self._operation = None
        self._cancel_event = None
        self._cancellation_control = None
        self._terminal_state = None
        self._gate.release(self)
        if worker is not None:
            worker.deleteLater()
        self.state_changed.emit()
        self._emit_settled_if_complete()
        if not self._close_fenced:
            if prerequisite_failed:
                if operation_tab_id is not None and reconcile_target_ids:
                    self.reconcile_canonical_targets(
                        operation_tab_id,
                        reconcile_target_ids,
                    )
                self.status_changed.emit(
                    "Preview baseline "
                    + ("cancelled" if summary.state == "cancelled" else "failed")
                    + "; draft and stable results were preserved."
                )
                if self._pending_interactive_preview is not None:
                    QtCore.QTimer.singleShot(
                        0,
                        self._redrive_pending_interactive_preview,
                    )
            elif operation_purpose == "preview-prerequisite":
                self.status_changed.emit(
                    "Preview baseline ready; starting exact Preview."
                )
                QtCore.QTimer.singleShot(0, self._redrive_pending_preview)
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
                if (
                    operation_tab_id is not None
                    and operation_purpose == "interactive-successor"
                    and interactive_successor_ids
                ):
                    repository = self._repository(operation_tab_id)
                    for set_id in interactive_successor_ids:
                        repository.clear_preview(set_id, clear_reference=False)
                    self.results_changed.emit(operation_tab_id)
                elif operation_tab_id is not None and reconcile_target_ids:
                    self.reconcile_canonical_targets(
                        operation_tab_id,
                        reconcile_target_ids,
                        clear_pending=False,
                    )
                self.status_changed.emit(self._terminal_status(summary))
                if self._pending_preview is not None:
                    QtCore.QTimer.singleShot(0, self._redrive_pending_preview)
                elif self._pending_interactive_preview is not None:
                    QtCore.QTimer.singleShot(
                        0,
                        self._redrive_pending_interactive_preview,
                    )
    @staticmethod
    def _terminal_status(summary: SimulationTerminalSummary) -> str:
        return f"{summary.kind.capitalize()} {summary.state}: {summary.succeeded} succeeded, {summary.failed} failed, {summary.cancelled} cancelled, {summary.discarded} discarded."

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
