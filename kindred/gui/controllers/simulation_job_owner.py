"""Concrete owner for one Run/Preview worker and latest pending Preview intent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from numbers import Real
import os
from threading import Event
from time import monotonic
from typing import Callable, Iterable, Literal

import numpy as np
from PySide6 import QtCore

from kindred.core.exceptions import SimulationCancelled
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
    canonical_initial_values,
)
from kindred.core.result_repository import (
    ExplicitResultCurrentFacts,
    ExplicitResultKey,
    PreviewResultIdentity,
    PreviewResultKey,
    ResultRepository,
    SimulationResultValue,
    explicit_result_key_from_current_facts,
    explicit_result_key_from_values,
    explicit_result_key_is_current,
    scientific_identity,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.simulation_preparation import (
    PreparedSimulationRun,
    SimulationExecutionRequest,
    build_simulation_execution_request,
    metadata_view_for_mechanism,
    prepare_simulation_worker_run,
    solver_config_for_execution_settings,
)
from kindred.core.simulation_result_finalization import (
    finalize_simulation_result,
)
from kindred.core.simulation_algebra_policy import SimulationAlgebraPolicy
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.temperature import temperature_schedule_execution_identity
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.simulator.wegscheider_symbolic import (
    WegscheiderCyclicityReport,
    analyze_wegscheider_cyclicity,
    validate_wegscheider_cyclicity_resolved,
)
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
    "PreviewIntent",
    "PreviewTarget",
    "SimulationJobOwner",
    "SimulationTerminalSummary",
    "RunParallelCapacity",
    "build_explicit_target",
    "build_preview_target",
    "resolve_run_parallel_capacity",
]


def _required_id(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


@dataclass(frozen=True, slots=True)
class RunParallelCapacity:
    cpu_budget: int
    default_user_max: int
    resolved_user_max: int
    effective: int


def resolve_run_parallel_capacity(
    *,
    cpu_count: object,
    user_max: object,
    target_count: int,
) -> RunParallelCapacity:
    """Resolve the complete Run capacity policy in its sole owner module."""
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("Run requires a positive exact target count.")
    targets = target_count
    cpu = (
        int(cpu_count)
        if type(cpu_count) is int and int(cpu_count) > 0
        else 1
    )
    cpu_budget = max(1, cpu - 1)
    default_user_max = min(cpu_budget, 16)
    resolved_user_max = (
        max(1, min(int(user_max), MAX_PARALLEL_WORKERS_CEILING))
        if type(user_max) is int
        else default_user_max
    )
    return RunParallelCapacity(
        cpu_budget=cpu_budget,
        default_user_max=default_user_max,
        resolved_user_max=resolved_user_max,
        effective=min(
            targets,
            cpu_budget,
            resolved_user_max,
            MAX_PARALLEL_WORKERS_CEILING,
        ),
    )


def _copied_execution_request(
    value: SimulationExecutionRequest,
) -> SimulationExecutionRequest:
    if isinstance(value, SimulationExecutionRequest):
        return value.isolated_copy()
    raise TypeError("execution_request must be SimulationExecutionRequest.")


def _validated_initial_values(
    values: Iterable[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    names: set[str] = set()
    for raw_name, raw_value in values:
        name = _required_id(raw_name, "initial value name")
        if name in names:
            raise ValueError(f"initial values contain duplicate name {name!r}.")
        if type(raw_value) is bool or not isinstance(raw_value, Real):
            raise TypeError(f"initial value for {name!r} must be numeric.")
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"initial value for {name!r} must be finite and nonnegative."
            )
        names.add(name)
        normalized.append((name, value))
    if not normalized:
        raise ValueError("a simulation target must contain initial values.")
    return tuple(normalized)


def _validate_request_against_result_key(
    request: SimulationExecutionRequest,
    key: ExplicitResultKey | PreviewResultKey,
    *,
    prepared_dsl: PreparedDslRequestSet,
) -> None:
    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
    analysis = request.snapshot.analysis
    if (
        getattr(analysis, "normalized_text", None)
        != key.mechanism_source.reactions_text.strip()
    ):
        raise ValueError(
            "execution request analysis does not match result source."
        )
    if getattr(analysis, "identity", None) != key.analysis_identity:
        raise ValueError(
            "execution request analysis identity does not match result key."
        )
    if tuple(
        zip(request.snapshot.species_names, request.snapshot.y0, strict=True)
    ) != key.initial_values:
        raise ValueError("execution request initials do not match result key.")
    expected_temperature_schedule = (
        metadata_view_for_mechanism(
            prepared_dsl.mechanism_for_execution()
        ).temperature_schedule
    )
    if temperature_schedule_execution_identity(
        request.snapshot.temperature_schedule
    ) != temperature_schedule_execution_identity(expected_temperature_schedule):
        raise ValueError(
            "execution request temperature schedule does not match "
            "the result key authority."
        )
    settings = key.execution_settings
    if tuple(request.t_span) != (settings.t_start, settings.t_end):
        raise ValueError("execution request interval does not match result key.")
    expected_solver_config = solver_config_for_execution_settings(
        execution_settings=settings,
        scientific_settings=key.scientific_settings,
    )
    if request.solver_config != expected_solver_config:
        raise ValueError(
            "execution request solver/scientific settings do not match "
            "result key."
        )
    if not isinstance(
        request.snapshot.intervention_execution,
        InterventionExecutionContract,
    ):
        raise ValueError(
            "execution request requires its bound intervention contract."
        )
    contract = request.snapshot.intervention_execution
    if isinstance(key, PreviewResultKey) and (
        key.intervention_declarative_fingerprint
        != contract.declarative_fingerprint
        or key.intervention_executable_fingerprint
        != contract.executable_fingerprint
    ):
        raise ValueError(
            "Preview key intervention identity does not match its compiled "
            "request contract."
        )


@dataclass(frozen=True, slots=True)
class ExplicitRunTarget:
    key: ExplicitResultKey
    set_name: str
    publication_generation: int
    prepared_dsl: PreparedDslRequestSet
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        if not isinstance(self.key, ExplicitResultKey):
            raise TypeError("ExplicitRunTarget key must be an ExplicitResultKey.")
        object.__setattr__(self, "set_name", _required_id(self.set_name, "set_name"))
        if type(self.publication_generation) is not int:
            raise TypeError("publication_generation must be an int.")
        generation = self.publication_generation
        if generation < 0:
            raise ValueError("publication_generation must be nonnegative.")
        object.__setattr__(self, "publication_generation", generation)
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError(
                "ExplicitRunTarget prepared_dsl must be PreparedDslRequestSet."
            )
        request = _copied_execution_request(self.execution_request)
        _validate_request_against_result_key(
            request,
            self.key,
            prepared_dsl=self.prepared_dsl,
        )
        contract = request.snapshot.intervention_execution
        if not isinstance(contract, InterventionExecutionContract):
            raise ValueError("Run target requires a compiled intervention contract.")
        if self.key.scientific_identity != scientific_identity(
            self.prepared_dsl,
            intervention_execution=contract,
        ):
            raise ValueError(
                "Run result key scientific identity does not match its exact "
                "compiled intervention contract."
            )
        if request.parameter_overrides:
            raise ValueError("explicit Run targets cannot carry Preview values.")
        object.__setattr__(self, "execution_request", request)


@dataclass(frozen=True, slots=True)
class ExplicitRunCapture:
    tab_id: str
    targets: tuple[ExplicitRunTarget, ...]

    def __post_init__(self) -> None:
        tab_id = _required_id(self.tab_id, "tab_id")
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("Run requires at least one exactly selected target.")
        set_ids: set[str] = set()
        for target in targets:
            if not isinstance(target, ExplicitRunTarget):
                raise TypeError("Run targets must be ExplicitRunTarget values.")
            if target.key.tab_id != tab_id:
                raise ValueError("every Run target must belong to the capture tab.")
            if target.key.set_id in set_ids:
                raise ValueError(f"Run target {target.key.set_id!r} is duplicated.")
            set_ids.add(target.key.set_id)
        object.__setattr__(self, "tab_id", tab_id)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class _PendingRunDetailedBalanceCorrection:
    generation: int
    correction: DetailedBalanceCorrectionRequest
    capture: ExplicitRunCapture

    def __post_init__(self) -> None:
        if int(self.generation) <= 0:
            raise ValueError("Detailed-balance generation must be positive.")
        if not isinstance(self.correction, DetailedBalanceCorrectionRequest):
            raise TypeError(
                "Run detailed-balance correction requires the editor contract."
            )
        if not isinstance(self.capture, ExplicitRunCapture):
            raise TypeError(
                "Run detailed-balance correction requires an exact Run capture."
            )


@dataclass(frozen=True, slots=True)
class PreviewTarget:
    key: PreviewResultKey
    set_name: str
    prepared_dsl: PreparedDslRequestSet
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        if not isinstance(self.key, PreviewResultKey):
            raise TypeError("PreviewTarget key must be a PreviewResultKey.")
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError(
                "PreviewTarget prepared_dsl must be PreparedDslRequestSet."
            )
        object.__setattr__(self, "set_name", _required_id(self.set_name, "set_name"))
        request = _copied_execution_request(self.execution_request)
        _validate_request_against_result_key(
            request,
            self.key,
            prepared_dsl=self.prepared_dsl,
        )
        request_parameter_values = tuple(
            (request.parameter_overrides or {}).items()
        )
        request_parameter_names = {
            name for name, _value in request_parameter_values
        }
        expected_worker_values = tuple(
            (name, value)
            for name, value in self.key.parameter_values
            if name in request_parameter_names
        )
        if request_parameter_values != expected_worker_values:
            raise ValueError(
                "execution request worker parameter values are not an exact "
                "subset of the Preview key."
            )
        object.__setattr__(self, "execution_request", request)


@dataclass(frozen=True, slots=True)
class PreviewIntent:
    tab_id: str
    targets: tuple[PreviewTarget, ...]

    def __post_init__(self) -> None:
        tab_id = _required_id(self.tab_id, "tab_id")
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("Preview requires at least one exact Slider target.")
        set_ids: set[str] = set()
        for target in targets:
            if not isinstance(target, PreviewTarget):
                raise TypeError("Preview targets must be PreviewTarget values.")
            if target.key.tab_id != tab_id:
                raise ValueError("every Preview target must belong to the intent tab.")
            if target.key.set_id in set_ids:
                raise ValueError(f"Preview target {target.key.set_id!r} is duplicated.")
            set_ids.add(target.key.set_id)
        object.__setattr__(self, "tab_id", tab_id)
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

    def __post_init__(self) -> None:
        if self.kind not in {"run", "preview"}:
            raise ValueError("terminal kind must be 'run' or 'preview'.")
        if self.state not in {"success", "failure", "cancelled", "partial"}:
            raise ValueError(
                "terminal state must be success, failure, cancelled, or partial."
            )


@dataclass(frozen=True, slots=True)
class _TargetTerminalOutcome:
    key: ExplicitResultKey | PreviewResultKey
    state: Literal["success", "failure", "cancelled"]
    stage: Literal["preparation", "execution"]
    result: SimulationResultValue | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, (ExplicitResultKey, PreviewResultKey)):
            raise TypeError(
                "terminal outcome key must be an explicit or Preview result key."
            )
        if self.state not in {"success", "failure", "cancelled"}:
            raise ValueError(
                "terminal outcome state must be success, failure, or cancelled."
            )
        if self.stage not in {"preparation", "execution"}:
            raise ValueError(
                "terminal outcome stage must be preparation or execution."
            )
        if self.state == "success":
            if not isinstance(self.result, SimulationResultValue):
                raise TypeError(
                    "a successful terminal outcome requires a simulation result."
                )
        elif self.result is not None:
            raise ValueError(
                "a failed or cancelled terminal outcome cannot carry a result."
            )
        object.__setattr__(self, "message", str(self.message or ""))


def build_explicit_target(
    *,
    prepared_dsl: PreparedDslRequestSet,
    tab_id: str,
    set_id: str,
    set_name: str,
    mechanism_source: MechanismAuthoringSource,
    initial_values: Iterable[tuple[str, float]],
    execution_settings: SimulationExecutionSettings,
    scientific_settings: TabScientificSettings,
    publication_generation: int,
) -> ExplicitRunTarget:
    values = _validated_initial_values(initial_values)
    execution_request = build_simulation_execution_request(
        prepared_dsl=prepared_dsl,
        mechanism_source=mechanism_source,
        initial_values=values,
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
    )
    intervention_execution = execution_request.snapshot.intervention_execution
    if not isinstance(intervention_execution, InterventionExecutionContract):
        raise ValueError("Run target intervention contract was not compiled.")
    key = explicit_result_key_from_values(
        tab_id=tab_id,
        set_id=set_id,
        mechanism_source=mechanism_source,
        prepared_dsl=prepared_dsl,
        analysis_identity=prepared_dsl.analysis.identity,
        initial_values=values,
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
        intervention_execution=intervention_execution,
    )
    return ExplicitRunTarget(
        key=key,
        set_name=set_name,
        publication_generation=publication_generation,
        prepared_dsl=prepared_dsl,
        execution_request=execution_request,
    )


def build_preview_target(
    *,
    prepared_dsl: PreparedDslRequestSet,
    key: PreviewResultKey,
    set_name: str,
    execution_request: SimulationExecutionRequest,
) -> PreviewTarget:
    return PreviewTarget(
        key=key,
        set_name=set_name,
        prepared_dsl=prepared_dsl,
        execution_request=execution_request,
    )


def _prepare_target(
    target: ExplicitRunTarget | PreviewTarget,
    *,
    cancellation_control: SimulationCancellationControl,
) -> tuple[
    PreparedSimulationRun,
    tuple[tuple[str, float, str], ...],
]:
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    request = target.execution_request
    parameters = _captured_parameter_coordinates(
        request,
        prepared_dsl=target.prepared_dsl,
    )
    prepared = prepare_simulation_worker_run(
        execution_request=request,
        cancellation_control=cancellation_control,
    )
    return prepared, parameters


def _captured_parameter_coordinates(
    request: SimulationExecutionRequest,
    *,
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[tuple[str, float, str], ...]:
    """Capture typed writable coordinate truth for one immutable request."""

    coordinates = enumerate_parameter_coordinates(
        prepared_dsl,
        wegscheider_cyclicity_enabled=(
            request.solver_config.wegscheider_cyclicity_enabled
        ),
    )
    by_name = {coordinate.name: coordinate for coordinate in coordinates}
    overrides = {
        str(name): float(value)
        for name, value in dict(request.parameter_overrides or {}).items()
    }
    for name in overrides:
        if name not in by_name:
            raise ValueError(f"Preview parameter {name!r} is not a direct GUI coordinate.")
    return tuple(
        (
            coordinate.name,
            overrides.get(coordinate.name, float(coordinate.value)),
            coordinate.unit,
        )
        for coordinate in coordinates
    )


def _detailed_balance_report(
    prepared_dsl: PreparedDslRequestSet,
) -> WegscheiderCyclicityReport:
    """Return typed cyclicity truth from one exact accepted source."""

    mechanism = prepared_dsl.mechanism_for_execution()
    parameter_algebra_spec = prepared_dsl.parameter_algebra_spec_for_execution()
    apply_parameter_algebra_spec_to_mechanism(
        parameter_algebra_spec,
        mechanism=mechanism,
        require_mutable=False,
    )
    return analyze_wegscheider_cyclicity(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
    )


def _source_energy_unit_for_prepared(prepared_dsl: PreparedDslRequestSet) -> str:
    """Return the accepted source energy unit carried by one DSL analysis."""

    return str(prepared_dsl.analysis.energy_unit)


def _detailed_balance_correction_for_capture(
    capture: ExplicitRunCapture,
) -> DetailedBalanceCorrectionRequest | None:
    unresolved: list[
        tuple[
            MechanismAuthoringSource,
            float,
            WegscheiderCyclicityReport,
            str,
        ]
    ] = []
    reports_by_analysis_identity: dict[
        str,
        WegscheiderCyclicityReport,
    ] = {}
    for target in capture.targets:
        if not solver_config_for_execution_settings(
            execution_settings=target.key.execution_settings,
            scientific_settings=target.key.scientific_settings,
        ).wegscheider_cyclicity_enabled:
            continue
        temperature = float(target.key.scientific_settings.temperature)
        analysis_identity = str(
            target.prepared_dsl.analysis.identity
        )
        report = reports_by_analysis_identity.get(analysis_identity)
        if report is None:
            report = _detailed_balance_report(target.prepared_dsl)
            reports_by_analysis_identity[analysis_identity] = report
        if report.unresolved_cycles:
            source_energy_unit = str(
                _source_energy_unit_for_prepared(target.prepared_dsl)
            )
            unresolved.append(
                (
                    target.key.mechanism_source,
                    temperature,
                    report,
                    source_energy_unit,
                )
            )
    if not unresolved:
        return None

    source, temperature, report, source_energy_unit = unresolved[0]
    cycle_truth = tuple(
        (
            cycle.cycle_id,
            cycle.step_indices,
            cycle.parameter_names,
        )
        for cycle in report.unresolved_cycles
    )
    if any(
        candidate_source != source
        or candidate_temperature != temperature
        or candidate_source_energy_unit != source_energy_unit
        or tuple(
            (
                cycle.cycle_id,
                cycle.step_indices,
                cycle.parameter_names,
            )
            for cycle in candidate_report.unresolved_cycles
        )
        != cycle_truth
        for (
            candidate_source,
            candidate_temperature,
            candidate_report,
            candidate_source_energy_unit,
        )
        in unresolved[1:]
    ):
        raise ValueError(
            "Run targets do not share one exact detailed-balance correction."
        )
    issue = "Unresolved detailed-balance cycle"
    if len(cycle_truth) != 1:
        issue += "s"
    issue += ": " + "; ".join(
        f"{cycle_id} ({', '.join(parameter_names)})"
        for cycle_id, _step_indices, parameter_names in cycle_truth
    )
    target_identity = "\0".join(
        f"{target.key.set_id}\0{target.publication_generation}"
        for target in capture.targets
    )
    request_id = hashlib.sha256(
        (
            f"Run\0{capture.tab_id}\0{temperature:.17g}\0"
            f"{source.reactions_text}\0{target_identity}\0"
            f"{report.symbolic_identity['source_fingerprint']}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return DetailedBalanceCorrectionRequest(
        request_id=request_id,
        origin="Run",
        tab_id=capture.tab_id,
        source=source.reactions_text,
        issue=issue,
        temperature_K=temperature,
        report=report,
        source_energy_unit=source_energy_unit,
    )


def _verify_detailed_balance_resolved(
    prepared_dsl: PreparedDslRequestSet,
) -> None:
    """Prove one accepted typed correction resolves every cycle."""

    mechanism = prepared_dsl.mechanism_for_execution()
    parameter_algebra_spec = prepared_dsl.parameter_algebra_spec_for_execution()
    apply_parameter_algebra_spec_to_mechanism(
        parameter_algebra_spec,
        mechanism=mechanism,
        require_mutable=False,
    )
    validate_wegscheider_cyclicity_resolved(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
    )


def _solve_prepared(
    prepared: PreparedSimulationRun,
    *,
    parameters: tuple[tuple[str, float, str], ...],
    cancellation_control: SimulationCancellationControl,
) -> SimulationResultValue:
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    solved = solve_ode(prepared.request)
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    result = finalize_simulation_result(
        mechanism=prepared.mechanism,
        result=solved,
        species_names=list(prepared.species_names),
        initials_for_algebra=prepared.initials_for_algebra,
        temperature_schedule=prepared.temperature_schedule,
        parameter_algebra_spec=prepared.parameter_algebra_spec,
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        parameters=parameters,
        preparation_warnings=prepared.warnings,
    )
    if cancellation_control.cancel_requested():
        raise SimulationCancelled()
    return result


class _DirectSimulationWorker(QtCore.QThread):
    target_terminal = QtCore.Signal(int, object)

    def __init__(
        self,
        *,
        job_id: int,
        targets: tuple[ExplicitRunTarget | PreviewTarget, ...],
        max_parallel: int,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._job_id = int(job_id)
        self._targets = tuple(targets)
        self._max_parallel = max(
            1,
            min(int(max_parallel), len(self._targets)),
        )
        self._cancelled = Event()
        self._cancellation_control = SimulationCancellationControl(
            self._cancelled.is_set
        )

    def cancel(self) -> None:
        self._cancelled.set()
        self.requestInterruption()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        with ThreadPoolExecutor(
            max_workers=self._max_parallel,
            thread_name_prefix="kindred-run",
        ) as pool:
            futures = {
                pool.submit(self._execute_target, target): target
                for target in self._targets
            }
            for future in as_completed(futures):
                try:
                    outcome = future.result()
                except BaseException as exc:  # noqa: BLE001
                    target = futures[future]
                    outcome = _TargetTerminalOutcome(
                        key=target.key,
                        state=(
                            "cancelled"
                            if self._cancelled.is_set()
                            else "failure"
                        ),
                        stage="preparation",
                        message=str(exc) or type(exc).__name__,
                    )
                self.target_terminal.emit(self._job_id, outcome)
                if self._cancelled.is_set():
                    for pending in futures:
                        pending.cancel()

    def _execute_target(
        self,
        target: ExplicitRunTarget | PreviewTarget,
    ) -> _TargetTerminalOutcome:
        if self._cancelled.is_set():
            return _TargetTerminalOutcome(
                key=target.key,
                state="cancelled",
                stage="preparation",
                message="Simulation cancelled before preparation.",
            )
        try:
            prepared, parameters = _prepare_target(
                target,
                cancellation_control=self._cancellation_control,
            )
        except SimulationCancelled:
            return _TargetTerminalOutcome(
                key=target.key,
                state="cancelled",
                stage="preparation",
                message="Simulation cancelled.",
            )
        except BaseException as exc:  # noqa: BLE001
            return _TargetTerminalOutcome(
                key=target.key,
                state="failure",
                stage="preparation",
                message=str(exc) or type(exc).__name__,
            )
        try:
            result = _solve_prepared(
                prepared,
                parameters=parameters,
                cancellation_control=self._cancellation_control,
            )
        except SimulationCancelled:
            return _TargetTerminalOutcome(
                key=target.key,
                state="cancelled",
                stage="execution",
                message="Simulation cancelled.",
            )
        except BaseException as exc:  # noqa: BLE001
            return _TargetTerminalOutcome(
                key=target.key,
                state="failure",
                stage="execution",
                message=str(exc) or type(exc).__name__,
            )
        return _TargetTerminalOutcome(
            key=target.key,
            state="success",
            stage="execution",
            result=result,
        )


RepositoryResolver = Callable[[str], ResultRepository | None]
CurrentExplicitFactsResolver = Callable[
    [str, str],
    ExplicitResultCurrentFacts | None,
]
RawUserParallelMaximum = Callable[[], object]


class SimulationJobOwner(QtCore.QObject):
    """Owns one worker, immutable captures, and publication generations."""

    state_changed = QtCore.Signal()
    owned_qthreads_changed = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    results_changed = QtCore.Signal(str)
    preview_endpoints_retired = QtCore.Signal(object)
    terminal = QtCore.Signal(object)
    run_dispatched = QtCore.Signal(str, object)
    detailed_balance_correction_requested = QtCore.Signal(object)

    def __init__(
        self,
        *,
        gate: ComputationGate,
        repository_for_tab: RepositoryResolver,
        current_explicit_result_facts: CurrentExplicitFactsResolver,
        propose_detailed_balance_source: Callable[
            [str, MechanismAuthoringSource],
            DetailedBalanceCorrectionProposal | None,
        ],
        commit_detailed_balance_source: Callable[
            [DetailedBalanceCorrectionProposal],
            DetailedBalanceCorrectionCommitResult,
        ],
        user_parallel_maximum: RawUserParallelMaximum | None = None,
        cpu_count: Callable[[], object] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(gate, ComputationGate):
            raise TypeError("gate must be a ComputationGate.")
        if not callable(repository_for_tab):
            raise TypeError("repository_for_tab must be callable.")
        if not callable(current_explicit_result_facts):
            raise TypeError("current_explicit_result_facts must be callable.")
        if not callable(propose_detailed_balance_source):
            raise TypeError("propose_detailed_balance_source must be callable.")
        if not callable(commit_detailed_balance_source):
            raise TypeError("commit_detailed_balance_source must be callable.")
        self._gate = gate
        self._repository_for_tab = repository_for_tab
        self._current_explicit_result_facts = current_explicit_result_facts
        self._propose_detailed_balance_source = propose_detailed_balance_source
        self._commit_detailed_balance_source = commit_detailed_balance_source
        self._user_parallel_maximum = user_parallel_maximum or (lambda: None)
        self._cpu_count = cpu_count or os.cpu_count
        self._worker: _DirectSimulationWorker | None = None
        self._job_id = 0
        self._job_kind: str | None = None
        self._origin_tab_id: str | None = None
        self._targets: tuple[ExplicitRunTarget | PreviewTarget, ...] = ()
        self._outcomes: dict[object, str] = {}
        self._details: list[str] = []
        self._pending_preview: PreviewIntent | None = None
        self._pending_detailed_balance: (
            _PendingRunDetailedBalanceCorrection | None
        ) = None
        self._detailed_balance_generation = 0
        self._explicit_generation_by_target: dict[tuple[str, str], int] = {}
        self._preview_generation_by_target: dict[tuple[str, str], int] = {}
        self._retire_on_settlement: set[tuple[str, str]] = set()
        self._current_preview_key_by_target: dict[
            tuple[str, str],
            PreviewResultKey,
        ] = {}
        self._application_close_fenced = False
        self._gate.changed.connect(self._on_gate_changed)

    @property
    def is_executing(self) -> bool:
        return bool(
            self._worker is not None
            or self._pending_detailed_balance is not None
        )

    def owned_qthreads(self) -> tuple[QtCore.QThread, ...]:
        """Return the exact native threads still retained by this owner."""

        worker = self._worker
        if worker is None:
            return ()
        return (worker,)

    @property
    def current_kind(self) -> str | None:
        if self._pending_detailed_balance is not None:
            return "run"
        return self._job_kind

    @property
    def origin_tab_id(self) -> str | None:
        pending = self._pending_detailed_balance
        if pending is not None:
            return pending.capture.tab_id
        return self._origin_tab_id

    @property
    def pending_preview_tab_id(self) -> str | None:
        return self._pending_preview.tab_id if self._pending_preview else None

    def is_tab_executing(self, tab_id: str) -> bool:
        target = str(tab_id or "")
        pending = self._pending_detailed_balance
        return bool(
            (self._worker is not None and self._origin_tab_id == target)
            or (pending is not None and pending.capture.tab_id == target)
        )

    def explicit_publication_generation(self, tab_id: str, set_id: str) -> int:
        target = (
            _required_id(tab_id, "tab_id"),
            _required_id(set_id, "set_id"),
        )
        return int(self._explicit_generation_by_target.get(target, 0))

    def advance_explicit_publication_generations(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]:
        """Fence affected in-flight Runs at the first accepted/transient edit."""

        tid = _required_id(tab_id, "tab_id")
        advanced: list[tuple[str, int]] = []
        seen: set[str] = set()
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            if sid in seen:
                continue
            seen.add(sid)
            target = (tid, sid)
            generation = int(self._explicit_generation_by_target.get(target, 0)) + 1
            self._explicit_generation_by_target[target] = generation
            advanced.append((sid, generation))
        return tuple(advanced)

    def _preview_publication_generation(self, tab_id: str, set_id: str) -> int:
        """Return the monotonic Preview generation tombstone for one target."""

        target = (
            _required_id(tab_id, "tab_id"),
            _required_id(set_id, "set_id"),
        )
        return int(self._preview_generation_by_target.get(target, 0))

    def _advance_preview_publication_generations(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]:
        """Fence Preview captures without deleting their monotonic tombstones."""

        tid = _required_id(tab_id, "tab_id")
        advanced: list[tuple[str, int]] = []
        seen: set[str] = set()
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            if sid in seen:
                continue
            seen.add(sid)
            target = (tid, sid)
            generation = int(
                self._preview_generation_by_target.get(target, 0)
            ) + 1
            self._preview_generation_by_target[target] = generation
            advanced.append((sid, generation))
        return tuple(advanced)

    def _cancel_generation_counter_retirement(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> None:
        tid = _required_id(tab_id, "tab_id")
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            self._retire_on_settlement.discard((tid, sid))

    def _settle_retired_generation_target(
        self,
        tab_id: str,
        set_id: str,
    ) -> None:
        target = (
            _required_id(tab_id, "tab_id"),
            _required_id(set_id, "set_id"),
        )
        if target not in self._retire_on_settlement:
            return
        self._explicit_generation_by_target.pop(target, None)
        self._preview_generation_by_target.pop(target, None)
        self._retire_on_settlement.discard(target)

    def _retire_generation_counters(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> None:
        tid = _required_id(tab_id, "tab_id")
        targets = {
            (tid, _required_id(set_id, "set_id")) for set_id in set_ids
        }
        awaiting_native_settlement = set(self._retire_on_settlement)
        awaiting_native_settlement.update(
            (target.key.tab_id, target.key.set_id)
            for target in self._targets
            if target.key not in self._outcomes
        )
        for target in targets:
            if target in awaiting_native_settlement:
                self._retire_on_settlement.add(target)
                continue
            self._explicit_generation_by_target.pop(target, None)
            self._preview_generation_by_target.pop(target, None)
            self._retire_on_settlement.discard(target)

    def _retire_all_generation_counters_for_tab(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> None:
        tid = _required_id(tab_id, "tab_id")
        targets = {_required_id(set_id, "set_id") for set_id in set_ids}
        targets.update(
            set_id
            for target_tab_id, set_id in self._explicit_generation_by_target
            if target_tab_id == tid
        )
        targets.update(
            set_id
            for target_tab_id, set_id in self._preview_generation_by_target
            if target_tab_id == tid
        )
        targets.update(
            set_id
            for target_tab_id, set_id in self._retire_on_settlement
            if target_tab_id == tid
        )
        self._retire_generation_counters(tid, targets)

    def preview_key_is_current(self, key: PreviewResultKey) -> bool:
        """Authorize publication only for the owner's exact latest Preview key."""

        if not isinstance(key, PreviewResultKey):
            return False
        target = (key.tab_id, key.set_id)
        return bool(
            self._current_preview_key_by_target.get(target) == key
            and key.generation
            == self._preview_publication_generation(
                key.tab_id,
                key.set_id,
            )
        )

    def _preview_target_is_active(self, tab_id: str, set_id: str) -> bool:
        target = (
            _required_id(tab_id, "tab_id"),
            _required_id(set_id, "set_id"),
        )
        return target in self._current_preview_key_by_target

    def _retire_preview_keys(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[PreviewResultKey, ...]:
        tid = _required_id(tab_id, "tab_id")
        retired: list[PreviewResultKey] = []
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            key = self._current_preview_key_by_target.pop(
                (tid, sid),
                None,
            )
            if key is not None:
                retired.append(key)
        if retired:
            self.preview_endpoints_retired.emit(tuple(retired))
        return tuple(retired)

    def _retire_all_preview_keys_for_tab(self, tab_id: str) -> None:
        tid = _required_id(tab_id, "tab_id")
        retired: list[PreviewResultKey] = []
        for target in tuple(self._current_preview_key_by_target):
            if target[0] == tid:
                key = self._current_preview_key_by_target.pop(target, None)
                if key is not None:
                    retired.append(key)
        if retired:
            self.preview_endpoints_retired.emit(tuple(retired))

    def _preview_key_is_pending(self, key: PreviewResultKey) -> bool:
        pending = self._pending_preview
        return bool(
            pending is not None
            and any(target.key == key for target in pending.targets)
        )

    def _retire_unsuccessful_preview_key(
        self,
        key: PreviewResultKey,
    ) -> bool:
        """Retire one failed Preview unless its exact intent is pending."""

        if not isinstance(key, PreviewResultKey):
            return False
        target = (key.tab_id, key.set_id)
        if self._current_preview_key_by_target.get(target) != key:
            return False
        if self._preview_key_is_pending(key):
            return False
        self._retire_preview_keys(key.tab_id, (key.set_id,))
        self._advance_preview_publication_generations(
            key.tab_id,
            (key.set_id,),
        )
        try:
            self._repository(key.tab_id).clear_preview(
                key.set_id,
                clear_reference=False,
            )
        except LookupError:
            pass
        self.results_changed.emit(key.tab_id)
        return True

    def _retire_unsuccessful_preview_intent(
        self,
        intent: PreviewIntent,
    ) -> None:
        for target in intent.targets:
            self._retire_unsuccessful_preview_key(target.key)

    def deauthorize_targets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
        *,
        preserve_reference: bool = False,
    ) -> tuple[tuple[str, int], ...]:
        """Atomically fence in-flight Run publication and current eligibility."""

        tid = _required_id(tab_id, "tab_id")
        targets = tuple(
            dict.fromkeys(_required_id(set_id, "set_id") for set_id in set_ids)
        )
        if not targets:
            return ()
        self._drop_pending_preview_for_targets(tid, targets)
        self._advance_preview_publication_generations(tid, targets)
        self._retire_preview_keys(tid, targets)
        generations = self.advance_explicit_publication_generations(tid, targets)
        repository = self._repository(tid)
        if preserve_reference:
            repository.deauthorize_sets(
                targets,
                preserve_reference=True,
            )
        else:
            repository.deauthorize_sets(targets)
        self.results_changed.emit(tid)
        return generations

    def begin_preview_edit(
        self,
        identity: PreviewResultIdentity,
    ) -> PreviewResultKey | None:
        """Install the exact Preview key after request construction.

        The caller fences the affected targets before constructing the
        request.  Reusing the already-advanced Preview generation keeps key
        installation from becoming a second, provisional currentness fence.
        """

        if not isinstance(identity, PreviewResultIdentity):
            raise TypeError("identity must be a PreviewResultIdentity.")
        if self._application_close_fenced:
            self.status_changed.emit("Preview rejected: simulation owner is closing.")
            return None
        target = (identity.tab_id, identity.set_id)
        generation = self._preview_publication_generation(
            identity.tab_id,
            identity.set_id,
        )
        if generation <= 0 or target in self._current_preview_key_by_target:
            raise RuntimeError(
                "Preview key installation requires a prior target fence."
            )
        key = PreviewResultKey.from_identity(
            identity,
            generation=generation,
        )
        repository = self._repository(identity.tab_id)
        self._drop_pending_preview_for_targets(identity.tab_id, (identity.set_id,))
        repository.capture_preview_reference(key)
        repository.authorize_preview_display(key)
        self._current_preview_key_by_target[
            (identity.tab_id, identity.set_id)
        ] = key
        self._cancel_generation_counter_retirement(
            identity.tab_id,
            (identity.set_id,),
        )
        self.results_changed.emit(identity.tab_id)
        return key

    def reconcile_canonical_targets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, bool], ...]:
        """Restore canonical targets from their retained exact Run keys."""

        tid = _required_id(tab_id, "tab_id")
        targets = tuple(
            dict.fromkeys(_required_id(set_id, "set_id") for set_id in set_ids)
        )
        if not targets:
            return ()
        repository = self._repository(tid)
        self._cancel_generation_counter_retirement(tid, targets)
        self._drop_pending_preview_for_targets(tid, targets)
        self._advance_preview_publication_generations(tid, targets)
        hydrated_by_set_id: dict[str, bool] = {}
        for set_id in targets:
            self._retire_preview_keys(tid, (set_id,))
            repository.clear_preview(
                set_id,
                clear_reference=False,
            )
            current_key = self._current_explicit_key(tid, set_id)
            hydrated = (
                None
                if current_key is None
                else repository.hydrate_cached(current_key)
            )
            hydrated_by_set_id[set_id] = hydrated is not None
            if hydrated is None:
                repository.clear_preview(set_id)
        self.results_changed.emit(tid)
        return tuple(
            (set_id, hydrated_by_set_id.get(set_id, False))
            for set_id in targets
        )

    def purge_sets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]:
        """Purge results and retire counters after native work settles."""
        tid = _required_id(tab_id, "tab_id")
        targets = tuple(
            dict.fromkeys(_required_id(value, "set_id") for value in set_ids)
        )
        if not targets:
            return ()
        self._drop_pending_preview_for_targets(tid, targets)
        target_set = set(targets)
        self._advance_preview_publication_generations(tid, targets)
        self._retire_preview_keys(tid, targets)
        pending_correction = self._pending_detailed_balance
        if (
            pending_correction is not None
            and pending_correction.capture.tab_id == tid
            and any(
                target.key.set_id in target_set
                for target in pending_correction.capture.targets
            )
        ):
            self._abort_pending_detailed_balance(
                "Run detailed-balance correction aborted: a target was deleted."
            )
        generations = self.advance_explicit_publication_generations(tid, targets)
        self._retire_generation_counters(tid, targets)
        self._discard_purged_admitted_targets(
            tid,
            target_set,
            reason="its set was purged and publication was deauthorized",
        )
        self._repository(tid).purge_sets(targets)
        self.results_changed.emit(tid)
        return generations

    def purge_tab(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]:
        """Fence a removed tab and retire counters after native settlement."""
        tid = _required_id(tab_id, "tab_id")
        targets = tuple(
            dict.fromkeys(_required_id(value, "set_id") for value in set_ids)
        )
        self.drop_pending_preview(tid)
        self._advance_preview_publication_generations(tid, targets)
        self._retire_all_preview_keys_for_tab(tid)
        pending_correction = self._pending_detailed_balance
        if (
            pending_correction is not None
            and pending_correction.capture.tab_id == tid
        ):
            self._abort_pending_detailed_balance(
                "Run detailed-balance correction aborted: its tab was deleted."
            )
        admitted_set_ids = {
            target.key.set_id
            for target in self._targets
            if target.key.tab_id == tid
        }
        deauthorized_set_ids = tuple(dict.fromkeys((*targets, *admitted_set_ids)))
        generations = self.advance_explicit_publication_generations(
            tid,
            deauthorized_set_ids,
        )
        self._retire_all_generation_counters_for_tab(
            tid,
            deauthorized_set_ids,
        )
        self._discard_purged_admitted_targets(
            tid,
            None,
            reason="its tab was purged and publication was deauthorized",
        )
        try:
            self._repository(tid).clear()
        except LookupError:
            pass
        self.results_changed.emit(tid)
        return generations

    def start_run(self, capture: ExplicitRunCapture) -> bool:
        if not isinstance(capture, ExplicitRunCapture):
            raise TypeError("capture must be an ExplicitRunCapture.")
        if self._application_close_fenced:
            self.status_changed.emit("Run rejected: simulation owner is closing.")
            return False
        if (
            self._worker is not None
            or self._pending_detailed_balance is not None
            or not self._gate.is_idle
        ):
            self.status_changed.emit("Run rejected: another computation is active.")
            return False
        for target in capture.targets:
            if (
                target.publication_generation
                != self.explicit_publication_generation(
                    target.key.tab_id,
                    target.key.set_id,
                )
                or not self._explicit_completion_is_live_key(target.key)
            ):
                self.status_changed.emit(
                    f"Run rejected: target {target.set_name!r} capture is stale."
                )
                return False
        try:
            correction = _detailed_balance_correction_for_capture(capture)
        except Exception as exc:
            self.status_changed.emit(
                "Run rejected by detailed-balance preanalysis: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        if correction is not None:
            if not self._gate.try_acquire(self, "run"):
                self.status_changed.emit(
                    "Run rejected: another computation is active."
                )
                return False
            self.drop_pending_preview(capture.tab_id)
            self._detailed_balance_generation += 1
            self._pending_detailed_balance = (
                _PendingRunDetailedBalanceCorrection(
                    generation=self._detailed_balance_generation,
                    correction=correction,
                    capture=capture,
                )
            )
            self.status_changed.emit(
                "Run requires a detailed-balance source correction before "
                "execution."
            )
            self.state_changed.emit()
            self.detailed_balance_correction_requested.emit(correction)
            return True
        return self._launch_run_capture(
            capture,
            gate_already_owned=False,
        )

    def _launch_run_capture(
        self,
        capture: ExplicitRunCapture,
        *,
        gate_already_owned: bool,
    ) -> bool:
        try:
            capacity = resolve_run_parallel_capacity(
                cpu_count=self._cpu_count(),
                user_max=self._user_parallel_maximum(),
                target_count=len(capture.targets),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.status_changed.emit(
                "Run rejected by parallel-capacity input: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        self.drop_pending_preview(capture.tab_id)
        try:
            started = self._start_worker(
                "run",
                capture.tab_id,
                capture.targets,
                max_parallel=capacity.effective,
                gate_already_owned=gate_already_owned,
            )
        except Exception as exc:
            self.status_changed.emit(
                f"Run dispatch failed: {type(exc).__name__}: {exc}"
            )
            return False
        if not started:
            self.status_changed.emit(
                "Run continuation rejected: its admission is no longer owned."
                if gate_already_owned
                else "Run rejected: another computation is active."
            )
            return False
        self.run_dispatched.emit(
            capture.tab_id,
            tuple(target.key.set_id for target in capture.targets),
        )
        self.status_changed.emit(
            f"Run started with capacity {capacity.effective}."
        )
        return True

    def complete_detailed_balance_correction(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> bool:
        """Preflight, commit, and continue one exact admitted Run once."""

        if not isinstance(outcome, DetailedBalanceCorrectionOutcome):
            raise TypeError("outcome must be DetailedBalanceCorrectionOutcome.")
        pending = self._pending_detailed_balance
        if pending is None:
            return False
        request = pending.correction
        if outcome.request_id != request.request_id:
            return False
        if (
            pending.generation != self._detailed_balance_generation
            or (
                outcome.origin,
                outcome.tab_id,
                outcome.source,
                outcome.issue,
            )
            != (
                request.origin,
                request.tab_id,
                request.source,
                request.issue,
            )
        ):
            self._abort_pending_detailed_balance(
                "Run detailed-balance correction aborted: request identity "
                "is no longer exact."
            )
            return False
        if outcome.cancelled:
            self._abort_pending_detailed_balance(
                "Detailed-balance correction cancelled; Run aborted."
            )
            return True
        corrected_text = outcome.corrected_source
        if corrected_text is None:
            raise ValueError("Accepted correction has no corrected source.")

        original_sources = {
            target.key.mechanism_source
            for target in pending.capture.targets
        }
        if len(original_sources) != 1:
            self._abort_pending_detailed_balance(
                "Run detailed-balance correction aborted: captured sources "
                "are no longer singular."
            )
            return False
        original_source = next(iter(original_sources))
        if original_source.reactions_text != request.source:
            self._abort_pending_detailed_balance(
                "Run detailed-balance correction aborted: captured source "
                "identity changed."
            )
            return False
        corrected_source = original_source.with_reactions_text(corrected_text)
        committed = False
        try:
            if (
                self._application_close_fenced
                or self._pending_detailed_balance is not pending
                or not self._gate.is_owned_by(self)
                or any(
                    not self._explicit_completion_is_live_key(target.key)
                    for target in pending.capture.targets
                )
            ):
                raise RuntimeError(
                    "Original Run capture is no longer the exact live "
                    "canonical request."
                )
            proposal = self._propose_detailed_balance_source(
                request.tab_id,
                corrected_source,
            )
            if not isinstance(proposal, DetailedBalanceCorrectionProposal):
                raise RuntimeError(
                    "Canonical source proposal rejected the correction."
                )
            self._build_detailed_balance_run_continuation(
                pending,
                proposal,
                bind_postcommit=False,
            )
            commit_result = self._commit_detailed_balance_source(proposal)
            if not isinstance(
                commit_result,
                DetailedBalanceCorrectionCommitResult,
            ):
                raise RuntimeError("Canonical source commit returned no result.")
            if commit_result.proposal != proposal:
                raise RuntimeError(
                    "Canonical source commit returned a different proposal."
                )
            if not commit_result.committed:
                raise RuntimeError(
                    "Canonical source commit rejected the correction"
                    + (
                        f": {commit_result.error}"
                        if commit_result.error
                        else "."
                    )
                )
            committed = True
            if commit_result.error:
                raise RuntimeError(
                    "Canonical source commit completed with a postcommit "
                    f"error: {commit_result.error}"
                )
            if (
                self._application_close_fenced
                or self._pending_detailed_balance is not pending
                or not self._gate.is_owned_by(self)
            ):
                raise RuntimeError(
                    "Run correction admission was retired after commit."
                )
            rebased = self._build_detailed_balance_run_continuation(
                pending,
                proposal,
                bind_postcommit=True,
            )
            self._pending_detailed_balance = None
            launched = self._launch_run_capture(
                rebased,
                gate_already_owned=True,
            )
            if not launched:
                if self._worker is not None:
                    # _start_worker() retains custody when native start state
                    # cannot be disproved.  Request cancellation, but never
                    # tear down or release the owner/gate before the native
                    # terminal signal settles that worker.
                    self._worker.cancel()
                    self.status_changed.emit(
                        "Detailed-balance correction committed; Run aborted "
                        "(native cancellation requested): worker startup "
                        "remains uncertain and owner-retained pending native "
                        "terminal delivery."
                    )
                    return False
                self._gate.release(self)
                raise RuntimeError("Corrected Run could not be dispatched.")
        except Exception as exc:
            if committed:
                if self._pending_detailed_balance is pending:
                    self._pending_detailed_balance = None
                if self._worker is None and self._gate.is_owned_by(self):
                    self._gate.release(self)
                self.status_changed.emit(
                    "Detailed-balance correction committed; Run aborted: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif self._pending_detailed_balance is pending:
                self._abort_pending_detailed_balance(
                    "Detailed-balance correction rejected before canonical "
                    "commit; source remains an uncommitted draft; Run aborted: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                self.status_changed.emit(
                    "Detailed-balance correction rejected before canonical "
                    "commit; source remains an uncommitted draft; Run aborted: "
                    f"{type(exc).__name__}: {exc}"
                )
            return False
        self.status_changed.emit(
            "Detailed-balance correction committed; continuing Run once."
        )
        return True

    def _build_detailed_balance_run_continuation(
        self,
        pending: _PendingRunDetailedBalanceCorrection,
        proposal: DetailedBalanceCorrectionProposal,
        *,
        bind_postcommit: bool,
    ) -> ExplicitRunCapture:
        """Validate, then bind corrected Run targets to canonical currentness.

        The proposal is the complete precommit validation source.  Once the
        canonical commit succeeds, that commit's mutation fence advances the
        captured Run generations, so the continuation is rebuilt from the
        live current-facts boundary before worker admission.
        """

        if proposal.tab_id != pending.capture.tab_id:
            raise RuntimeError(
                "Detailed-balance proposal tab does not match the pending Run."
            )
        proposed_tab = proposal.proposed_document.tab(proposal.tab_id)
        corrected_prepared = proposed_tab.prepared_dsl
        if not isinstance(corrected_prepared, PreparedDslRequestSet):
            raise RuntimeError("Proposed Run source has no retained prepared DSL.")
        _verify_detailed_balance_resolved(corrected_prepared)
        rebased_targets: list[ExplicitRunTarget] = []
        for target in pending.capture.targets:
            proposed_set = proposed_tab.initial_condition_set(target.key.set_id)
            initial_values = canonical_initial_values(
                proposed_tab,
                target.key.set_id,
            )
            corrected_settings = proposed_tab.resolved_execution_settings(
                target.key.set_id
            )
            corrected_request = build_simulation_execution_request(
                prepared_dsl=corrected_prepared,
                mechanism_source=proposed_tab.mechanism_source,
                initial_values=initial_values,
                execution_settings=corrected_settings,
                scientific_settings=proposed_tab.scientific_settings,
            )
            corrected_contract = corrected_request.snapshot.intervention_execution
            if not isinstance(corrected_contract, InterventionExecutionContract):
                raise RuntimeError(
                    "Corrected Run target has no compiled intervention contract."
                )
            corrected_key = explicit_result_key_from_values(
                tab_id=target.key.tab_id,
                set_id=target.key.set_id,
                mechanism_source=proposed_tab.mechanism_source,
                prepared_dsl=corrected_prepared,
                analysis_identity=corrected_prepared.analysis.identity,
                initial_values=initial_values,
                execution_settings=corrected_settings,
                scientific_settings=proposed_tab.scientific_settings,
                intervention_execution=corrected_contract,
            )
            target_key = corrected_key
            target_prepared = corrected_prepared
            target_source = proposed_tab.mechanism_source
            target_initial_values = initial_values
            target_execution_settings = corrected_settings
            target_scientific_settings = proposed_tab.scientific_settings
            target_request = corrected_request
            if bind_postcommit:
                facts = self._current_explicit_result_facts(
                    target.key.tab_id,
                    target.key.set_id,
                )
                if not isinstance(facts, ExplicitResultCurrentFacts):
                    raise RuntimeError(
                        f"Corrected Run target {target.set_name!r} has no "
                        "canonical postcommit current facts."
                    )
                current_key = explicit_result_key_from_current_facts(facts)
                if current_key != corrected_key:
                    raise RuntimeError(
                        f"Corrected Run target {target.set_name!r} is not the "
                        "exact canonical postcommit target."
                    )
                if not isinstance(facts.prepared_dsl, PreparedDslRequestSet):
                    raise RuntimeError(
                        f"Corrected Run target {target.set_name!r} has no "
                        "canonical postcommit prepared DSL."
                    )
                target_key = current_key
                target_prepared = facts.prepared_dsl
                target_source = facts.mechanism_source
                target_initial_values = facts.initial_values
                target_execution_settings = facts.execution_settings
                target_scientific_settings = facts.scientific_settings
                target_request = build_simulation_execution_request(
                    prepared_dsl=target_prepared,
                    mechanism_source=target_source,
                    initial_values=target_initial_values,
                    execution_settings=target_execution_settings,
                    scientific_settings=target_scientific_settings,
                )
            rebased_targets.append(
                ExplicitRunTarget(
                    key=target_key,
                    set_name=proposed_set.name,
                    publication_generation=(
                        self.explicit_publication_generation(
                            target_key.tab_id,
                            target_key.set_id,
                        )
                    ),
                    prepared_dsl=target_prepared,
                    execution_request=target_request,
                )
            )
        return ExplicitRunCapture(
            tab_id=pending.capture.tab_id,
            targets=tuple(rebased_targets),
        )

    def _abort_pending_detailed_balance(self, detail: str) -> bool:
        pending = self._pending_detailed_balance
        if pending is None:
            return False
        self._pending_detailed_balance = None
        self._gate.release(self)
        self.status_changed.emit(str(detail))
        self.state_changed.emit()
        return True

    def request_preview(self, intent: PreviewIntent) -> bool:
        if not isinstance(intent, PreviewIntent):
            raise TypeError("intent must be a PreviewIntent.")
        if self._application_close_fenced:
            self._retire_unsuccessful_preview_intent(intent)
            self.status_changed.emit("Preview rejected: simulation owner is closing.")
            return False
        current_intent = self._current_preview_intent(intent)
        if current_intent is None:
            self.status_changed.emit("Preview rejected: every exact key is stale.")
            return False
        intent = current_intent
        if self._worker is not None or not self._gate.is_idle:
            # This is the one latest debounce intent, not admitted work. It never
            # cancels, preempts, or expands into a retained queue.
            self._pending_preview = intent
            self.status_changed.emit(
                "Preview not admitted while another computation is active; "
                "latest intent retained."
            )
            self.state_changed.emit()
            return False
        try:
            started = self._start_worker(
                "preview",
                intent.tab_id,
                intent.targets,
                max_parallel=1,
                gate_already_owned=False,
            )
        except Exception as exc:
            self._retire_unsuccessful_preview_intent(intent)
            self.status_changed.emit(
                f"Preview dispatch failed: {type(exc).__name__}: {exc}"
            )
            return False
        if not started:
            self._pending_preview = intent
            self.status_changed.emit(
                "Preview not admitted while another computation is active; "
                "latest intent retained."
            )
            self.state_changed.emit()
            return False
        self.status_changed.emit("Preview started.")
        return True

    def drop_pending_preview(self, tab_id: str | None = None) -> None:
        if self._pending_preview is None:
            return
        if tab_id is not None and self._pending_preview.tab_id != str(tab_id or ""):
            return
        self._pending_preview = None
        self.state_changed.emit()

    def _drop_pending_preview_for_targets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> None:
        pending = self._pending_preview
        if pending is None or pending.tab_id != str(tab_id):
            return
        targets = {str(value) for value in set_ids}
        if not any(
            target.key.set_id in targets
            for target in pending.targets
        ):
            return
        surviving = tuple(
            target
            for target in pending.targets
            if target.key.set_id not in targets
        )
        self._pending_preview = (
            PreviewIntent(tab_id=pending.tab_id, targets=surviving)
            if surviving
            else None
        )
        self.state_changed.emit()

    def cancel_current(self) -> bool:
        if self._pending_detailed_balance is not None:
            return self._abort_pending_detailed_balance(
                "Run cancellation requested while detailed-balance correction "
                "was pending."
            )
        worker = self._worker
        if worker is None:
            return False
        worker.cancel()
        detail = (
            f"{(self._job_kind or 'simulation').capitalize()} "
            "cancellation requested."
        )
        self._details.append(detail)
        self.status_changed.emit(detail)
        return True

    def fence_application_close_admission(self) -> None:
        """Reject Run and Preview admission for the active Close attempt."""

        if self._application_close_fenced:
            return
        self._application_close_fenced = True
        self.drop_pending_preview()
        self._abort_pending_detailed_balance(
            "Run detailed-balance correction aborted during Close."
        )
        self.state_changed.emit()

    def cancel_for_application_close(self) -> None:
        """Request cancellation without waiting for native thread exit."""

        self.fence_application_close_admission()
        worker = self._worker
        if worker is not None:
            worker.cancel()

    def cancel_and_wait_for_project_replacement(self, deadline: float) -> bool:
        """Blocking cleanup solely for an accepted project replacement."""

        self.drop_pending_preview()
        self._abort_pending_detailed_balance(
            "Run detailed-balance correction aborted by project replacement."
        )
        worker = self._worker
        if worker is None:
            return True
        worker.cancel()
        remaining_ms = max(0, int((float(deadline) - monotonic()) * 1000.0))
        if not worker.wait(remaining_ms):
            return False
        self._finish_after_synchronous_wait()
        return True

    def show_cached(self, tab_id: str, set_id: str) -> bool:
        tid = _required_id(tab_id, "tab_id")
        sid = _required_id(set_id, "set_id")
        repository = self._repository(tid)
        current_key = self._current_explicit_key(tid, sid)
        if (
            current_key is None
            or self._preview_target_is_active(tid, sid)
        ):
            self.status_changed.emit(
                "Requested result is not currently display-eligible; display unchanged."
            )
            return False
        if repository.hydrate_cached(current_key) is None:
            self.status_changed.emit("Requested result is not cached; display unchanged.")
            return False
        self.results_changed.emit(tid)
        self.status_changed.emit("Cached result loaded for display.")
        return True

    def _start_worker(
        self,
        kind: str,
        tab_id: str,
        targets: tuple[ExplicitRunTarget | PreviewTarget, ...],
        *,
        max_parallel: int,
        gate_already_owned: bool,
    ) -> bool:
        """Acquire/install atomically; roll back only a verified unstarted worker."""

        if self._worker is not None:
            return False
        worker: _DirectSimulationWorker | None = None
        installed = False
        try:
            if gate_already_owned:
                if not self._gate.is_owned_by(self):
                    return False
            elif not self._gate.try_acquire(self, kind):
                return False
            next_job_id = self._job_id + 1
            admitted_targets = tuple(targets)
            if kind == "run":
                if any(
                    not isinstance(target, ExplicitRunTarget)
                    or not self._explicit_completion_is_current(target)
                    for target in admitted_targets
                ):
                    raise RuntimeError(
                        "Run targets changed during worker admission."
                    )
            elif kind == "preview":
                admitted_targets = tuple(
                    target
                    for target in admitted_targets
                    if isinstance(target, PreviewTarget)
                    and self.preview_key_is_current(target.key)
                )
                if not admitted_targets:
                    raise RuntimeError(
                        "Every Preview target changed during worker admission."
                    )
            else:
                raise ValueError("worker kind must be 'run' or 'preview'.")
            worker = _DirectSimulationWorker(
                job_id=next_job_id,
                targets=admitted_targets,
                max_parallel=max_parallel,
                parent=self,
            )
            worker.target_terminal.connect(self._on_target_terminal)
            worker.finished.connect(self._on_worker_finished)
            self._job_id = next_job_id
            self._job_kind = kind
            self._origin_tab_id = tab_id
            self._targets = admitted_targets
            self._outcomes = {}
            self._details = []
            self._worker = worker
            installed = True
            worker.start()
            self._cancel_generation_counter_retirement(
                tab_id,
                (target.key.set_id for target in admitted_targets),
            )
        except BaseException as exc:  # noqa: BLE001
            if installed and worker is not None:
                try:
                    verified_unstarted = bool(
                        not worker.isRunning()
                        and not worker.isFinished()
                    )
                except BaseException:  # noqa: BLE001
                    verified_unstarted = False
                if not verified_unstarted:
                    self.owned_qthreads_changed.emit()
                    self.state_changed.emit()
                    raise RuntimeError(
                        f"{kind.capitalize()} worker startup is uncertain; "
                        "native execution began or could not be disproved, so "
                        "the worker and computation gate remain owner-retained."
                    ) from exc
            if worker is not None:
                worker.cancel()
                try:
                    worker.target_terminal.disconnect(self._on_target_terminal)
                except (RuntimeError, TypeError):
                    pass
                try:
                    worker.finished.disconnect(self._on_worker_finished)
                except (RuntimeError, TypeError):
                    pass
            if installed and self._worker is worker:
                self._worker = None
                self._job_kind = None
                self._origin_tab_id = None
                self._targets = ()
                self._outcomes = {}
                self._details = []
            self._gate.release(self)
            if worker is not None:
                worker.deleteLater()
            if installed:
                self.owned_qthreads_changed.emit()
                self.state_changed.emit()
            raise
        self.owned_qthreads_changed.emit()
        self.state_changed.emit()
        return True

    @QtCore.Slot(int, object)
    def _on_target_terminal(
        self,
        job_id: int,
        outcome: object,
    ) -> None:
        if (
            int(job_id) != self._job_id
            or self._worker is None
            or not isinstance(outcome, _TargetTerminalOutcome)
        ):
            return
        key = outcome.key
        if self._target_for_key(key) is None:
            return
        self._settle_retired_generation_target(key.tab_id, key.set_id)
        if key in self._outcomes:
            return
        if outcome.state != "success":
            target_name = self._target_display_name(key)
            detail = (
                f"{target_name} {outcome.stage} {outcome.state}: "
                f"{outcome.message}"
            )
            self._record_terminal(key, outcome.state, detail)
            return
        result = outcome.result
        if not isinstance(result, SimulationResultValue):
            self._record_terminal(
                key,
                "failure",
                f"{self._target_display_name(key)} execution failed: "
                "worker returned no admitted result.",
            )
            return
        if self._worker.cancelled or self._application_close_fenced:
            self._record_terminal(
                key,
                "cancelled",
                f"{self._target_display_name(key)} cancelled before publication.",
            )
            return
        if isinstance(key, ExplicitResultKey) and self._job_kind == "run":
            target = self._target_for_key(key)
            if not isinstance(target, ExplicitRunTarget):
                self._record_discard(
                    key,
                    "Run result discarded: its captured target is unavailable.",
                )
                return
            if not self._explicit_completion_is_current(target):
                self._record_discard(
                    key,
                    f"Run result discarded for {target.set_name}: "
                    "the exact live target or publication generation changed.",
                )
                return
            try:
                repository = self._repository(key.tab_id)
            except LookupError:
                self._record_discard(
                    key,
                    f"Run result discarded for {target.set_name}: "
                    "the live result repository is unavailable.",
                )
                return
            if self._worker.cancelled or self._application_close_fenced:
                self._record_terminal(
                    key,
                    "cancelled",
                    f"Run result for {target.set_name} cancelled before publication.",
                )
                return
            try:
                repository.put_explicit(key, result)
            except Exception as exc:
                self._record_discard(
                    key,
                    f"Run result discarded for {target.set_name}: "
                    f"repository rejected publication ({type(exc).__name__}: {exc}).",
                )
                return
            if self._preview_target_is_active(key.tab_id, key.set_id):
                self._advance_preview_publication_generations(
                    key.tab_id,
                    (key.set_id,),
                )
                self._retire_preview_keys(key.tab_id, (key.set_id,))
            self._record_terminal(key, "success")
            self.results_changed.emit(key.tab_id)
            return
        if isinstance(key, PreviewResultKey) and self._job_kind == "preview":
            target_name = self._target_display_name(key)
            if not self.preview_key_is_current(key):
                self._record_discard(
                    key,
                    f"Preview result discarded for {target_name}: "
                    "the owner generation or exact live key changed.",
                )
                return
            try:
                repository = self._repository(key.tab_id)
            except LookupError:
                self._record_discard(
                    key,
                    f"Preview result discarded for {target_name}: "
                    "the live result repository is unavailable.",
                )
                return
            if self._worker.cancelled or self._application_close_fenced:
                self._record_terminal(
                    key,
                    "cancelled",
                    f"Preview result for {target_name} cancelled before publication.",
                )
                return
            try:
                repository.store_preview(key, result)
            except Exception as exc:
                self._record_discard(
                    key,
                    f"Preview result discarded for {target_name}: "
                    f"repository rejected publication ({type(exc).__name__}: {exc}).",
                )
                return
            self._record_terminal(key, "success")
            # Semantic publication is complete before any render attempt,
            # including for inactive tabs and non-plotted targets.
            self.results_changed.emit(key.tab_id)
            return
        self._record_discard(
            key,
            "Simulation result discarded: its job kind no longer matches.",
        )

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        if self.sender() is not self._worker:
            return
        self._settle_missing_terminal_outcomes()
        summary = self._terminal_summary()
        self._finish_current()
        if self._application_close_fenced:
            return
        self.terminal.emit(summary)
        self.status_changed.emit(self._terminal_status(summary))
        self._schedule_pending_preview()

    def _terminal_summary(self) -> SimulationTerminalSummary:
        kind = self._job_kind or "run"
        succeeded = sum(value == "success" for value in self._outcomes.values())
        failed = sum(value == "failure" for value in self._outcomes.values())
        cancelled = sum(value == "cancelled" for value in self._outcomes.values())
        discarded = sum(value == "discarded" for value in self._outcomes.values())
        details = list(self._details)
        total = succeeded + failed + cancelled + discarded
        if total and succeeded == total:
            state = "success"
        elif total and failed == total:
            state = "failure"
        elif total and cancelled == total:
            state = "cancelled"
        else:
            state = "partial"
        return SimulationTerminalSummary(
            kind=kind,
            state=state,
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            discarded=discarded,
            details=tuple(details),
        )

    @staticmethod
    def _terminal_status(summary: SimulationTerminalSummary) -> str:
        status = (
            f"{summary.kind.capitalize()} {summary.state}: "
            f"{summary.succeeded} succeeded, {summary.failed} failed, "
            f"{summary.cancelled} cancelled, {summary.discarded} discarded."
        )
        if summary.details:
            status = f"{status} Details: {'; '.join(summary.details)}"
        return status

    def _finish_after_synchronous_wait(self) -> None:
        self._settle_missing_terminal_outcomes()
        summary = self._terminal_summary()
        self._job_id += 1
        self._finish_current()
        self.terminal.emit(summary)
        self.status_changed.emit(self._terminal_status(summary))

    def _finish_current(self) -> None:
        worker = self._worker
        for target in self._targets:
            self._settle_retired_generation_target(
                target.key.tab_id,
                target.key.set_id,
            )
        self._worker = None
        self._job_kind = None
        self._origin_tab_id = None
        self._targets = ()
        self._outcomes = {}
        self._details = []
        self._gate.release(self)
        if worker is not None:
            worker.deleteLater()
            self.owned_qthreads_changed.emit()
        self.state_changed.emit()

    def _schedule_pending_preview(self) -> None:
        if self._pending_preview is not None:
            QtCore.QTimer.singleShot(0, self._redrive_pending_preview)

    @QtCore.Slot()
    def _redrive_pending_preview(self) -> None:
        if (
            self._application_close_fenced
            or self._worker is not None
            or self._pending_preview is None
            or not self._gate.is_idle
        ):
            return
        intent = self._pending_preview
        self._pending_preview = None
        current_intent = self._current_preview_intent(intent)
        if current_intent is not None:
            self.request_preview(current_intent)
        else:
            self.status_changed.emit(
                "Pending Preview discarded: every exact key changed."
            )
            self.state_changed.emit()

    @QtCore.Slot()
    def _on_gate_changed(self) -> None:
        if self._gate.is_idle:
            self._schedule_pending_preview()

    def _repository(self, tab_id: str) -> ResultRepository:
        repository = self._repository_for_tab(str(tab_id or ""))
        if not isinstance(repository, ResultRepository):
            raise LookupError(f"No ResultRepository exists for tab {tab_id!r}.")
        return repository

    def _target_for_key(
        self,
        key: ExplicitResultKey | PreviewResultKey,
    ) -> ExplicitRunTarget | PreviewTarget | None:
        return next((target for target in self._targets if target.key == key), None)

    def _target_display_name(self, key: object) -> str:
        if not isinstance(key, (ExplicitResultKey, PreviewResultKey)):
            return "target"
        target = self._target_for_key(key)
        return target.set_name if target is not None else "target"

    def _discard_purged_admitted_targets(
        self,
        tab_id: str,
        set_ids: set[str] | None,
        *,
        reason: str,
    ) -> None:
        """Deauthorize purged work without mutating its admitted roster."""

        worker = self._worker
        if worker is None:
            return
        for target in self._targets:
            key = target.key
            if key.tab_id != tab_id:
                continue
            if set_ids is not None and key.set_id not in set_ids:
                continue
            self._record_discard(
                key,
                f"{target.set_name} discarded: {reason}.",
            )
        if not any(
            target.key not in self._outcomes
            and self._admitted_target_is_publishable(target)
            for target in self._targets
        ):
            worker.cancel()

    def _admitted_target_is_publishable(
        self,
        target: ExplicitRunTarget | PreviewTarget,
    ) -> bool:
        if self._application_close_fenced:
            return False
        if isinstance(target, ExplicitRunTarget) and self._job_kind == "run":
            return self._explicit_completion_is_current(target)
        if isinstance(target, PreviewTarget) and self._job_kind == "preview":
            return self.preview_key_is_current(target.key)
        return False

    def _record_discard(self, key: object, detail: str) -> bool:
        return self._record_terminal(key, "discarded", detail)

    def _record_terminal(
        self,
        key: object,
        state: str,
        detail: str | None = None,
    ) -> bool:
        if state not in {"success", "failure", "cancelled", "discarded"}:
            raise ValueError(f"unsupported target terminal state {state!r}.")
        if key in self._outcomes:
            return False
        if not isinstance(key, (ExplicitResultKey, PreviewResultKey)):
            raise TypeError("a target terminal outcome requires a result key.")
        if self._target_for_key(key) is None:
            raise RuntimeError(
                "a terminal outcome must belong to the immutable admitted roster."
            )
        if isinstance(key, PreviewResultKey) and state != "success":
            self._retire_unsuccessful_preview_key(key)
        self._outcomes[key] = state
        if detail is not None:
            message = str(detail)
            self._details.append(message)
            self.status_changed.emit(message)
        return True

    def _settle_missing_terminal_outcomes(self) -> None:
        """Close every target exactly once if a worker ended without a signal."""

        worker_cancelled = bool(
            self._worker is not None and self._worker.cancelled
        )
        for target in self._targets:
            key = target.key
            if key in self._outcomes:
                continue
            state = "cancelled" if worker_cancelled else "failure"
            self._record_terminal(
                key,
                state,
                f"{target.set_name} {state}: worker ended without a terminal signal.",
            )

    def _explicit_completion_is_current(self, target: ExplicitRunTarget) -> bool:
        key = target.key
        if target.publication_generation != self.explicit_publication_generation(
            key.tab_id,
            key.set_id,
        ):
            return False
        return self._explicit_completion_is_live_key(key)

    def _explicit_completion_is_live_key(self, key: ExplicitResultKey) -> bool:
        try:
            current_key = self._current_explicit_key(key.tab_id, key.set_id)
            return explicit_result_key_is_current(key, current_key)
        except Exception:
            return False

    def _current_explicit_key(
        self,
        tab_id: str,
        set_id: str,
    ) -> ExplicitResultKey | None:
        """Resolve one current exact key from the shell's typed capture."""

        try:
            facts = self._current_explicit_result_facts(
                _required_id(tab_id, "tab_id"),
                _required_id(set_id, "set_id"),
            )
            if facts is None:
                return None
            return explicit_result_key_from_current_facts(facts)
        except Exception:
            return None

    def _current_preview_intent(
        self,
        intent: PreviewIntent,
    ) -> PreviewIntent | None:
        current_targets = tuple(
            target
            for target in intent.targets
            if self.preview_key_is_current(target.key)
        )
        if not current_targets:
            return None
        if current_targets == intent.targets:
            return intent
        return PreviewIntent(tab_id=intent.tab_id, targets=current_targets)
