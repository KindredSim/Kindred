"""Direct Run and exact Preview lifecycle owner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from numbers import Real
import os
from threading import Event
from time import monotonic
from typing import Callable, Iterable, Literal

import numpy as np
from PySide6 import QtCore

from kindred.core.exceptions import SimulationCancelled
from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
)
from kindred.core.result_repository import ResultRepository, SimulationResultValue
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.simulation_algebra_policy import SimulationAlgebraPolicy
from kindred.core.simulation_failure import FactualCause, factual_cause_from_exception
from kindred.core.simulation_preparation import (
    PreparedSimulationRun,
    SimulationExecutionRequest,
    SimulationPreparationError,
    build_simulation_execution_request,
    prepare_simulation_worker_run,
)
from kindred.core.simulation_result_finalization import finalize_simulation_result
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.workflow_contracts import (
    DetailedBalanceCorrectionCommitResult,
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionProposal,
    DetailedBalanceCorrectionRequest,
)

__all__ = [
    "ExplicitRunCapture",
    "ExplicitRunTarget",
    "PreviewTarget",
    "RunParallelCapacity",
    "SimulationJobOwner",
    "SimulationTerminalSummary",
    "build_explicit_target",
    "resolve_run_parallel_capacity",
]


def _required_id(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be nonblank.")
    return result


@dataclass(frozen=True, slots=True)
class RunParallelCapacity:
    resolved_user_max: int
    effective: int


def resolve_run_parallel_capacity(*, cpu_count: object, user_max: object, target_count: int) -> RunParallelCapacity:
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("Run requires a positive exact target count.")
    cpu = int(cpu_count) if type(cpu_count) is int and cpu_count > 0 else 1
    budget = max(1, cpu - 1)
    default = min(budget, 16)
    resolved = max(1, min(int(user_max), MAX_PARALLEL_WORKERS_CEILING)) if type(user_max) is int else default
    return RunParallelCapacity(resolved, min(target_count, budget, resolved, MAX_PARALLEL_WORKERS_CEILING))


def _copied_execution_request(value: SimulationExecutionRequest) -> SimulationExecutionRequest:
    if not isinstance(value, SimulationExecutionRequest):
        raise TypeError("execution_request must be SimulationExecutionRequest.")
    return value.isolated_copy()


def _validated_initial_values(values: Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    names: set[str] = set()
    for raw_name, raw_value in values:
        name = _required_id(raw_name, "initial value name")
        if name in names:
            raise ValueError(f"initial values contain duplicate name {name!r}.")
        if type(raw_value) is bool or not isinstance(raw_value, Real):
            raise TypeError(f"initial value for {name!r} must be numeric.")
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"initial value for {name!r} must be finite and nonnegative.")
        names.add(name)
        result.append((name, value))
    if not result:
        raise ValueError("a simulation target must contain initial values.")
    return tuple(result)


def _validate_execution_request(request: SimulationExecutionRequest, *, prepared_dsl: PreparedDslRequestSet) -> None:
    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
    if request.analysis != prepared_dsl.analysis:
        raise ValueError("execution request analysis does not match retained prepared DSL.")
    if not isinstance(request.intervention_execution, InterventionExecutionContract):
        raise ValueError("execution request requires its bound intervention contract.")


@dataclass(frozen=True, slots=True)
class ExplicitRunTarget:
    tab_id: str
    set_id: str
    set_name: str
    prepared_dsl: PreparedDslRequestSet
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(self, "set_name", _required_id(self.set_name, "set_name"))
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
        request = _copied_execution_request(self.execution_request)
        _validate_execution_request(request, prepared_dsl=self.prepared_dsl)
        if request.parameter_overrides:
            raise ValueError("explicit Run targets cannot carry Preview values.")
        object.__setattr__(self, "execution_request", request)


@dataclass(frozen=True, slots=True)
class ExplicitRunCapture:
    tab_id: str
    targets: tuple[ExplicitRunTarget, ...]

    def __post_init__(self) -> None:
        tab = _required_id(self.tab_id, "tab_id")
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
    prepared_dsl: PreparedDslRequestSet
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(self, "set_name", _required_id(self.set_name, "set_name"))
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
        request = _copied_execution_request(self.execution_request)
        _validate_execution_request(request, prepared_dsl=self.prepared_dsl)
        object.__setattr__(self, "execution_request", request)


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
class _PendingRunDetailedBalanceCorrection:
    generation: int
    correction: DetailedBalanceCorrectionRequest
    capture: ExplicitRunCapture


@dataclass(frozen=True, slots=True)
class _TargetTerminalOutcome:
    target: ExplicitRunTarget | PreviewTarget
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


def _format_owner_failure_cause(cause: FactualCause) -> str:
    return str(cause.message or "simulation failed")


def build_explicit_target(*, prepared_dsl: PreparedDslRequestSet, tab_id: str, set_id: str, set_name: str,
                          mechanism_source: MechanismAuthoringSource, initial_values: Iterable[tuple[str, float]],
                          execution_settings: SimulationExecutionSettings, scientific_settings: TabScientificSettings) -> ExplicitRunTarget:
    execution_request = build_simulation_execution_request(
        prepared_dsl=prepared_dsl, mechanism_source=mechanism_source,
        initial_values=_validated_initial_values(initial_values),
        execution_settings=execution_settings, scientific_settings=scientific_settings,
    )
    return ExplicitRunTarget(tab_id, set_id, set_name, prepared_dsl, execution_request)


def _captured_parameter_coordinates(request: SimulationExecutionRequest, *, prepared_dsl: PreparedDslRequestSet) -> tuple[tuple[str, float, str], ...]:
    coordinates = enumerate_parameter_coordinates(prepared_dsl, wegscheider_cyclicity_enabled=request.solver_config.wegscheider_cyclicity_enabled)
    by_name = {coordinate.name: coordinate for coordinate in coordinates}
    overrides = {str(name): float(value) for name, value in dict(request.parameter_overrides or {}).items()}
    for name in overrides:
        if name not in by_name:
            raise ValueError(f"Preview parameter {name!r} is not a direct GUI coordinate.")
    return tuple((coordinate.name, overrides.get(coordinate.name, float(coordinate.value)), coordinate.unit) for coordinate in coordinates)


def _prepare_target(target: ExplicitRunTarget | PreviewTarget, *, cancellation_control: SimulationCancellationControl) -> tuple[PreparedSimulationRun, tuple[tuple[str, float, str], ...]]:
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    parameters = _captured_parameter_coordinates(target.execution_request, prepared_dsl=target.prepared_dsl)
    prepared = prepare_simulation_worker_run(execution_request=target.execution_request, cancellation_control=cancellation_control)
    return prepared, parameters


def _solve_prepared(prepared: PreparedSimulationRun, *, parameters: tuple[tuple[str, float, str], ...], cancellation_control: SimulationCancellationControl) -> SimulationResultValue:
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    solved = solve_ode(prepared.request)
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

    def __init__(self, *, generation: int, targets: tuple[ExplicitRunTarget | PreviewTarget, ...],
                 max_parallel: int, cancellation_control: SimulationCancellationControl,
                 cancel_event: Event, parent: QtCore.QObject) -> None:
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

    def _execute_target(self, target: ExplicitRunTarget | PreviewTarget) -> _TargetTerminalOutcome:
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
CurrentRunCaptureResolver = Callable[
    [str, tuple[str, ...]], ExplicitRunCapture | None
]
class SimulationJobOwner(QtCore.QObject):
    """Own Run and exact Preview, which share the direct worker boundary."""

    state_changed = QtCore.Signal()
    application_close_settlement_changed = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    results_changed = QtCore.Signal(str)
    run_dispatched = QtCore.Signal(str, object)
    detailed_balance_correction_requested = QtCore.Signal(object)

    def __init__(self, *, gate: ComputationGate, repository_for_tab: RepositoryResolver,
                 propose_detailed_balance_source: Callable[[str, MechanismAuthoringSource], DetailedBalanceCorrectionProposal | None],
                 commit_detailed_balance_source: Callable[[DetailedBalanceCorrectionProposal], DetailedBalanceCorrectionCommitResult],
                 capture_current_run: CurrentRunCaptureResolver,
                 user_parallel_maximum: RawUserParallelMaximum | None = None,
                 cpu_count: Callable[[], object] | None = None,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        if not isinstance(gate, ComputationGate):
            raise TypeError("gate must be a ComputationGate.")
        if not callable(repository_for_tab) or not callable(propose_detailed_balance_source) or not callable(commit_detailed_balance_source):
            raise TypeError("Simulation owner callbacks must be callable.")
        self._gate = gate
        self._repository_for_tab = repository_for_tab
        self._propose_detailed_balance_source = propose_detailed_balance_source
        self._commit_detailed_balance_source = commit_detailed_balance_source
        if not callable(capture_current_run):
            raise TypeError("capture_current_run must be callable.")
        self._capture_current_run = capture_current_run
        self._user_parallel_maximum = user_parallel_maximum or (lambda: None)
        self._cpu_count = cpu_count or os.cpu_count
        self._generation = 0
        self._operation: dict[str, object] | None = None
        self._worker: _DirectSimulationWorker | None = None
        self._cancel_event: Event | None = None
        self._cancellation_control: SimulationCancellationControl | None = None
        self._terminal_state: str | None = None
        self._pending_preview: tuple[str, tuple[PreviewTarget, ...]] | None = None
        self._pending_detailed_balance: _PendingRunDetailedBalanceCorrection | None = None
        self._close_fenced = False
        self._last_close_settled = True
        self._gate.changed.connect(self._on_gate_changed)

    @property
    def application_close_settled(self) -> bool:
        return bool(
            self._worker is None
            and self._operation is None
            and self._pending_preview is None
            and self._pending_detailed_balance is None
        )

    @property
    def current_kind(self) -> str | None:
        if self._pending_detailed_balance is not None:
            return "run"
        if self._operation is not None:
            return str(self._operation["kind"])
        return "preview" if self._pending_preview is not None else None

    @property
    def origin_tab_id(self) -> str | None:
        if self._pending_detailed_balance is not None:
            return self._pending_detailed_balance.capture.tab_id
        operation = self._operation
        if operation is not None:
            return str(operation["tab_id"])
        return None if self._pending_preview is None else self._pending_preview[0]

    @property
    def pending_preview_tab_id(self) -> str | None:
        return None if self._pending_preview is None else self._pending_preview[0]

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
        pending_correction = self._pending_detailed_balance if include_run else None
        if (
            pending_correction is not None
            and pending_correction.capture.tab_id == tab
            and ids.intersection(
                target.set_id for target in pending_correction.capture.targets
            )
        ):
            self._generation += 1
            self._pending_detailed_balance = None
            self._terminal_state = "cancelled"
            if self._gate.is_owned_by(self):
                self._gate.release(self)
            return True
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
        if pending is None or pending[0] != str(tab_id):
            return False
        ids = {
            _required_id(value, "set_id")
            for value in set_ids
        }
        if not ids.intersection(target.set_id for target in pending[1]):
            return False
        self._pending_preview = None
        return True

    def deauthorize_targets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
        *,
        preserve_reference: bool = False,
    ) -> None:
        ids = tuple(dict.fromkeys(_required_id(value, "set_id") for value in set_ids))
        if not ids:
            return
        self._retire_if_affected(tab_id, ids)
        pending_changed = self._retire_pending_preview_if_affected(tab_id, ids)
        repository = self._repository(tab_id)
        if preserve_reference:
            repository.deauthorize_sets(ids, preserve_reference=True)
        else:
            repository.deauthorize_sets(ids)
        self.results_changed.emit(str(tab_id))
        if pending_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()

    def reconcile_canonical_targets(self, tab_id: str, set_ids: Iterable[str]) -> tuple[tuple[str, bool], ...]:
        ids = tuple(dict.fromkeys(_required_id(value, "set_id") for value in set_ids))
        self._retire_if_affected(tab_id, ids, include_run=False)
        pending_changed = self._retire_pending_preview_if_affected(tab_id, ids)
        repository = self._repository(tab_id)
        restored: list[tuple[str, bool]] = []
        for set_id in ids:
            repository.clear_preview(set_id, clear_reference=False)
            restored_value = repository.restore_reference_or_current(set_id)
            restored.append((set_id, restored_value))
        self.results_changed.emit(str(tab_id))
        if pending_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return tuple(restored)

    def purge_sets(self, tab_id: str, set_ids: Iterable[str]) -> None:
        ids = tuple(dict.fromkeys(_required_id(value, "set_id") for value in set_ids))
        self._retire_if_affected(tab_id, ids)
        pending_changed = self._retire_pending_preview_if_affected(tab_id, ids)
        self._repository(tab_id).purge_sets(ids)
        self.results_changed.emit(str(tab_id))
        if pending_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()

    def purge_tab(self, tab_id: str, set_ids: Iterable[str]) -> None:
        tab = _required_id(tab_id, "tab_id")
        ids = tuple(_required_id(value, "set_id") for value in set_ids)
        self._retire_if_affected(tab, ids)
        pending_changed = self._pending_preview is not None and self._pending_preview[0] == tab
        if pending_changed:
            self._pending_preview = None
        try:
            self._repository(tab).clear()
        except LookupError:
            pass
        self.results_changed.emit(tab)
        if pending_changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()

    def start_run(self, capture: ExplicitRunCapture) -> bool:
        if not isinstance(capture, ExplicitRunCapture):
            raise TypeError("capture must be ExplicitRunCapture.")
        if self._close_fenced or self._worker is not None or self._pending_detailed_balance is not None or not self._gate.is_idle:
            self.status_changed.emit("Run rejected: another computation is active.")
            return False
        correction = self._detailed_balance_correction(capture)
        if correction is not None:
            if not self._gate.try_acquire(self, "run"):
                return False
            self._mark_unsettled()
            self._generation += 1
            self._pending_detailed_balance = _PendingRunDetailedBalanceCorrection(self._generation, correction, capture)
            self.detailed_balance_correction_requested.emit(correction)
            self.state_changed.emit()
            return True
        return self._start(capture.tab_id, "run", capture.targets, max_parallel=self._capacity(len(capture.targets)))

    def _capacity(self, count: int) -> int:
        return resolve_run_parallel_capacity(cpu_count=self._cpu_count(), user_max=self._user_parallel_maximum(), target_count=count).effective

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

    def _start(self, tab_id: str, kind: str, targets: tuple[ExplicitRunTarget | PreviewTarget, ...], *, max_parallel: int) -> bool:
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
        self._operation = {"generation": generation, "kind": kind, "tab_id": str(tab_id), "targets": targets, "outcomes": [], "details": []}
        self._worker = worker
        self._cancel_event = event
        self._cancellation_control = control
        self._terminal_state = None
        worker.target_terminal.connect(self._on_target_terminal)
        worker.finished.connect(self._on_worker_finished)
        try:
            worker.start()
            if kind == "preview":
                self._promote_preview_targets(
                    str(tab_id),
                    tuple(targets),
                )
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
        self.state_changed.emit()
        if kind == "run":
            self.run_dispatched.emit(str(tab_id), tuple(target.set_id for target in targets))
        return True

    @staticmethod
    def _build_preview_targets(
        tab_id: str,
        requests: Iterable[
            tuple[str, str, SimulationExecutionRequest]
        ],
        *,
        prepared_dsl: PreparedDslRequestSet,
    ) -> tuple[PreviewTarget, ...]:
        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
        captured = tuple(requests)
        if not captured:
            raise ValueError("Preview requires at least one exact request.")
        tab = _required_id(tab_id, "tab_id")
        targets: list[PreviewTarget] = []
        seen: set[str] = set()
        for item in captured:
            if type(item) is not tuple or len(item) != 3:
                raise TypeError(
                    "Preview requests must be (set_id, set_name, request) tuples."
                )
            set_id, set_name, request = item
            sid = _required_id(set_id, "set_id")
            if sid in seen:
                raise ValueError(f"Preview target {sid!r} is duplicated.")
            if not isinstance(request, SimulationExecutionRequest):
                raise TypeError(
                    "Preview requests must contain SimulationExecutionRequest values."
                )
            seen.add(sid)
            targets.append(
                PreviewTarget(
                    tab_id=tab,
                    set_id=sid,
                    set_name=_required_id(set_name, "set_name"),
                    prepared_dsl=prepared_dsl,
                    execution_request=request,
                )
            )
        return tuple(targets)

    def request_preview(
        self,
        tab_id: str,
        requests: Iterable[
            tuple[str, str, SimulationExecutionRequest]
        ],
        *,
        prepared_dsl: PreparedDslRequestSet,
    ) -> bool:
        tab = _required_id(tab_id, "tab_id")
        captured = self._build_preview_targets(
            tab,
            requests,
            prepared_dsl=prepared_dsl,
        )
        if self._close_fenced:
            return False
        self._pending_preview = (tab, captured)
        self._mark_unsettled()
        if self._worker is not None or not self._gate.is_idle:
            self.status_changed.emit("Preview pending: latest exact request retained.")
            self.state_changed.emit()
            return False
        return self._redrive_pending_preview()

    def _redrive_pending_preview(self) -> bool:
        pending = self._pending_preview
        if pending is None or self._worker is not None or not self._gate.is_idle or self._close_fenced:
            return False
        # Submission is intent only.  Fence intersecting active work at the
        # start boundary, immediately before semantic Preview promotion.
        self._retire_if_affected(
            pending[0],
            (target.set_id for target in pending[1]),
        )
        self._pending_preview = None
        started = self._start(pending[0], "preview", pending[1], max_parallel=1)
        if not started:
            self._pending_preview = pending
        return started

    def retire_preview(
        self,
        tab_id: str,
        set_ids: Iterable[str] | None = None,
    ) -> bool:
        """Retire only an affected exact Preview before a newer gesture."""

        tab = _required_id(tab_id, "tab_id")
        ids = None if set_ids is None else tuple(
            dict.fromkeys(_required_id(value, "set_id") for value in set_ids)
        )
        changed = False
        operation = self._operation
        if (
            operation is not None
            and str(operation["kind"]) == "preview"
            and self._pending_detailed_balance is None
        ):
            operation_ids = tuple(
                target.set_id for target in operation["targets"]
            )
            affected = operation_ids if ids is None else ids
            changed = self._retire_if_affected(
                tab,
                affected,
                include_run=False,
            )
        pending = self._pending_preview
        if pending is not None and pending[0] == tab:
            pending_ids = tuple(target.set_id for target in pending[1])
            affected = pending_ids if ids is None else ids
            pending_matches = bool(
                set(affected).intersection(pending_ids)
            )
            if pending_matches:
                self._pending_preview = None
                changed = True
        if changed:
            self.state_changed.emit()
            self._emit_settled_if_complete()
        return changed

    def cancel_current(self) -> bool:
        changed = bool(self._worker is not None or self._pending_preview is not None or self._pending_detailed_balance is not None)
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
        self._pending_preview = None
        if self._pending_detailed_balance is not None:
            self._pending_detailed_balance = None
            if self._gate.is_owned_by(self):
                self._gate.release(self)
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

    def request_project_replacement_cancellation(self) -> None:
        self.cancel_current()

    def settle_project_replacement(self, deadline: float) -> bool:
        while not self.application_close_settled:
            worker = self._worker
            if worker is None:
                break
            remaining = max(0.0, float(deadline) - monotonic())
            if remaining <= 0.0:
                return False
            worker.wait(int(min(remaining, 0.05) * 1000))
            QtCore.QCoreApplication.processEvents()
        return self.application_close_settled

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
        # The one direct publication gate: compare owner operation generation
        # immediately before touching the canonical repository.
        if self._close_fenced:
            operation["outcomes"].append((target, "discarded"))
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
        if operation is not None:
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
            summary = SimulationTerminalSummary(str(operation["kind"]), state, succeeded, failed, cancelled, discarded, tuple(operation["details"]))
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
        if not self._close_fenced:
            self.status_changed.emit(self._terminal_status(summary))
            QtCore.QTimer.singleShot(0, self._redrive_pending_preview)
        self._emit_settled_if_complete()

    @staticmethod
    def _terminal_status(summary: SimulationTerminalSummary) -> str:
        return f"{summary.kind.capitalize()} {summary.state}: {summary.succeeded} succeeded, {summary.failed} failed, {summary.cancelled} cancelled, {summary.discarded} discarded."

    @QtCore.Slot()
    def _on_gate_changed(self) -> None:
        if self._gate.is_idle and self._worker is None and self._pending_preview is not None:
            QtCore.QTimer.singleShot(0, self._redrive_pending_preview)

    def _detailed_balance_correction(self, capture: ExplicitRunCapture) -> DetailedBalanceCorrectionRequest | None:
        unresolved: list[tuple[MechanismAuthoringSource, float, WegscheiderCyclicityReport, str]] = []
        for target in capture.targets:
            request = target.execution_request
            if not request.solver_config.wegscheider_cyclicity_enabled:
                continue
            report = target.prepared_dsl.wegscheider_admission_report()
            if report.unresolved_cycles:
                source = request.mechanism_source
                if not isinstance(source, MechanismAuthoringSource):
                    raise ValueError("Run detailed-balance correction requires retained source.")
                unresolved.append((source, float(request.analysis.temperature_K), report, str(target.prepared_dsl.analysis.energy_unit)))
        if not unresolved:
            return None
        source, temperature, report, unit = unresolved[0]
        cycles = tuple((cycle.cycle_id, cycle.step_indices, cycle.parameter_names) for cycle in report.unresolved_cycles)
        return DetailedBalanceCorrectionRequest(
            origin="Run", tab_id=capture.tab_id, source=source.reactions_text,
            issue="Unresolved detailed-balance cycle(s): " + "; ".join(cycle_id for cycle_id, _steps, _names in cycles),
            temperature_K=temperature, report=report, source_energy_unit=unit,
            accepted_analysis=capture.targets[0].prepared_dsl.analysis,
        )

    def _fail_detailed_balance_continuation(self, message: str) -> bool:
        """Close a committed correction when current Run capture is unavailable."""

        self._pending_detailed_balance = None
        self._terminal_state = "failure"
        if self._gate.is_owned_by(self):
            self._gate.release(self)
        self.status_changed.emit(str(message))
        self.state_changed.emit()
        self._emit_settled_if_complete()
        return False

    def complete_detailed_balance_correction(self, outcome: DetailedBalanceCorrectionOutcome) -> bool:
        pending = self._pending_detailed_balance
        if pending is None:
            return False
        if not isinstance(outcome, DetailedBalanceCorrectionOutcome) or outcome.cancelled:
            self._pending_detailed_balance = None
            self._gate.release(self)
            self.state_changed.emit()
            return True
        if outcome.source != pending.correction.source or outcome.issue != pending.correction.issue:
            self._pending_detailed_balance = None
            self._gate.release(self)
            return False
        source = pending.capture.targets[0].execution_request.mechanism_source
        if not isinstance(source, MechanismAuthoringSource) or outcome.corrected_source is None:
            return False
        proposal = self._propose_detailed_balance_source(pending.capture.tab_id, source.with_reactions_text(outcome.corrected_source))
        if not isinstance(proposal, DetailedBalanceCorrectionProposal):
            return False
        commit = self._commit_detailed_balance_source(proposal)
        if not isinstance(commit, DetailedBalanceCorrectionCommitResult) or not commit.committed:
            return False
        resolver = self._capture_current_run
        requested_ids = tuple(target.set_id for target in pending.capture.targets)
        try:
            fresh_capture = resolver(pending.capture.tab_id, requested_ids)
        except Exception as exc:  # noqa: BLE001
            return self._fail_detailed_balance_continuation(
                "Run correction committed, but current Run inputs are invalid: "
                f"{exc}"
            )
        if not isinstance(fresh_capture, ExplicitRunCapture):
            return self._fail_detailed_balance_continuation(
                "Run correction committed, but current Run inputs could not be rebuilt."
            )
        fresh_ids = tuple(target.set_id for target in fresh_capture.targets)
        if (
            fresh_capture.tab_id != pending.capture.tab_id
            or fresh_ids != requested_ids
        ):
            return self._fail_detailed_balance_continuation(
                "Run correction committed, but selected Run inputs are no longer current."
            )
        try:
            if self._detailed_balance_correction(fresh_capture) is not None:
                return self._fail_detailed_balance_continuation(
                    "Run correction committed, but corrected Run inputs remain invalid."
                )
        except Exception as exc:  # noqa: BLE001
            return self._fail_detailed_balance_continuation(
                "Run correction committed, but corrected Run inputs are invalid: "
                f"{exc}"
            )
        self._pending_detailed_balance = None
        if self._gate.is_owned_by(self):
            self._gate.release(self)
        return self._start(
            fresh_capture.tab_id,
            "run",
            fresh_capture.targets,
            max_parallel=self._capacity(len(fresh_capture.targets)),
        )
