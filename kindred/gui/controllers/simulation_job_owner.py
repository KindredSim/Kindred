"""Concrete owner for one Run/Preview worker and latest pending Preview intent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
from numbers import Real
import os
from threading import Event
from time import monotonic
from typing import Callable, Iterable, Mapping

import numpy as np
from PySide6 import QtCore

from kindred.core.exceptions import SimulationCancelled
from kindred.core.intervention_schedule import coerce_intervention_schedule
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)
from kindred.core.result_repository import (
    ExplicitResultKey,
    PreviewResultIdentity,
    PreviewResultKey,
    ResultRepository,
    SimulationResultProvenance,
    SimulationResultValue,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.simulation_preparation import (
    PreparedSimulationRun,
    SimulationExecutionRequest,
    prepare_simulation_worker_run,
)
from kindred.core.simulation_result_finalization import (
    evaluate_simulation_result_algebra,
)
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
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
from kindred.gui.widgets.mechanism_editor import (
    DetailedBalanceCorrectionOutcome,
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
    "build_execution_request",
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
    value: SimulationExecutionRequest | Mapping[str, object],
) -> SimulationExecutionRequest:
    if isinstance(value, SimulationExecutionRequest):
        return SimulationExecutionRequest.from_mapping(value.to_payload())
    if isinstance(value, Mapping):
        return SimulationExecutionRequest.from_mapping(value)
    raise TypeError(
        "execution_request must be a SimulationExecutionRequest or mapping."
    )


def _policy_enabled(value: object, *, field: str) -> bool:
    if type(value) is bool:
        return bool(value)
    raw = getattr(value, "value", value)
    normalized = str(raw or "").strip().casefold().replace("_", "-")
    if normalized in {"enabled", "enable", "on", "true", "yes", "auto"}:
        return True
    if normalized in {"disabled", "disable", "off", "false", "no", "never"}:
        return False
    raise ValueError(f"{field} has unsupported value {raw!r}.")


def _solver_config_for_settings(
    *,
    execution_settings: SimulationExecutionSettings,
    scientific_settings: TabScientificSettings,
) -> dict[str, object]:
    return {
        "solver": str(execution_settings.solver),
        "rtol": float(scientific_settings.rtol),
        "atol": float(scientific_settings.atol),
        "grid": {"N": int(execution_settings.integration_points)},
        "use_sparse_jacobian": _policy_enabled(
            scientific_settings.sparse_jacobian_policy,
            field="sparse_jacobian_policy",
        ),
        MechanismMetadataKeys.TEMPERATURE_K: float(
            scientific_settings.temperature
        ),
        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: _policy_enabled(
            scientific_settings.wegscheider_policy,
            field="wegscheider_policy",
        ),
    }


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


def _validated_parameter_values(
    values: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if not values:
        return None
    normalized: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = _required_id(raw_name, "parameter value name")
        if name in normalized:
            raise ValueError(
                f"parameter values contain duplicate name {name!r}."
            )
        if type(raw_value) is bool or not isinstance(raw_value, Real):
            raise TypeError(f"parameter value for {name!r} must be numeric.")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(
                f"parameter value for {name!r} must be finite."
            )
        normalized[name] = value
    return normalized


def build_execution_request(
    *,
    prepared_dsl: PreparedDslRequestSet,
    mechanism_source: MechanismAuthoringSource,
    initial_values: Iterable[tuple[str, float]],
    execution_settings: SimulationExecutionSettings,
    scientific_settings: TabScientificSettings,
    parameter_values: Mapping[str, float] | None = None,
) -> SimulationExecutionRequest:
    """Capture exactly four execution fields plus separate tab science."""

    if not isinstance(mechanism_source, MechanismAuthoringSource):
        raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    if not isinstance(execution_settings, SimulationExecutionSettings):
        raise TypeError(
            "execution_settings must be a SimulationExecutionSettings value."
        )
    validate_simulation_execution_settings(execution_settings)
    validate_tab_scientific_settings(scientific_settings)
    initials = dict(_validated_initial_values(initial_values))

    t_start = float(execution_settings.t_start)
    t_end = float(execution_settings.t_end)
    points = int(execution_settings.integration_points)
    if (
        not np.isfinite(t_start)
        or not np.isfinite(t_end)
        or t_start < 0.0
        or t_end <= t_start
    ):
        raise ValueError(
            "execution settings require finite 0 <= t_start < t_end."
        )
    if not 10 <= points <= 100_000:
        raise ValueError("integration_points must be between 10 and 100000.")
    parameters = _validated_parameter_values(parameter_values)
    solver_config = _solver_config_for_settings(
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
    )
    if (
        prepared_dsl.analysis.normalized_text
        != mechanism_source.full_dsl.strip()
    ):
        raise ValueError(
            "prepared DSL source does not match the captured mechanism source."
        )
    if (
        float(prepared_dsl.analysis.structure["temperature_K"])
        != float(scientific_settings.temperature)
    ):
        raise ValueError(
            "prepared DSL nominal temperature does not match tab science."
        )
    intervention_schedule = (
        prepared_dsl.intervention_schedule_for_execution()
    )
    prepared_payload = prepared_dsl.process_execution_payload(
        wegscheider_cyclicity_enabled=bool(
            scientific_settings.wegscheider_policy
        )
    )
    return SimulationExecutionRequest(
        prepared_payload=prepared_payload,
        initials=initials,
        t_span=(t_start, t_end),
        solver_config=solver_config,
        parameter_overrides=parameters,
        intervention_schedule=intervention_schedule,
    )


def _validate_request_against_result_key(
    request: SimulationExecutionRequest,
    key: ExplicitResultKey | PreviewResultKey,
) -> None:
    if request.prepared_payload is None or request.simulation_identity:
        raise ValueError(
            "execution request requires one retained prepared DSL authority."
        )
    analysis = request.prepared_payload.get("dsl_analysis")
    if (
        getattr(analysis, "normalized_text", None)
        != key.mechanism_source.full_dsl.strip()
    ):
        raise ValueError(
            "execution request analysis does not match result source."
        )
    if getattr(analysis, "identity", None) != key.analysis_identity:
        raise ValueError(
            "execution request analysis identity does not match result key."
        )
    if tuple(request.initials.items()) != key.initial_values:
        raise ValueError("execution request initials do not match result key.")
    settings = key.execution_settings
    if tuple(request.t_span) != (settings.t_start, settings.t_end):
        raise ValueError("execution request interval does not match result key.")
    actual_solver_config = dict(request.solver_config)
    raw_grid = actual_solver_config.get("grid")
    if isinstance(raw_grid, Mapping):
        actual_solver_config["grid"] = dict(raw_grid)
    expected_solver_config = _solver_config_for_settings(
        execution_settings=settings,
        scientific_settings=key.scientific_settings,
    )
    if actual_solver_config != expected_solver_config:
        raise ValueError(
            "execution request solver/scientific settings do not match "
            "result key."
        )
    expected_schedule = coerce_intervention_schedule(
        request.prepared_payload.get("unresolved_intervention_schedule")
    )
    if (
        not request.has_intervention_schedule_authority
        or request.intervention_schedule != expected_schedule
    ):
        raise ValueError(
            "execution request intervention schedule does not match source."
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
        _validate_request_against_result_key(request, self.key)
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
        _validate_request_against_result_key(request, self.key)
        if tuple((request.parameter_overrides or {}).items()) != (
            self.key.parameter_values
        ):
            raise ValueError(
                "execution request parameter values do not match Preview key."
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
class _TargetExecutionOutcome:
    key: ExplicitResultKey | PreviewResultKey
    accepted: bool
    cancelled: bool
    result: SimulationResultValue | None = None
    message: str = ""


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
    key = ExplicitResultKey(
        tab_id=tab_id,
        set_id=set_id,
        mechanism_source=mechanism_source,
        analysis_identity=prepared_dsl.analysis.identity,
        initial_values=values,
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
    )
    return ExplicitRunTarget(
        key=key,
        set_name=set_name,
        publication_generation=publication_generation,
        prepared_dsl=prepared_dsl,
        execution_request=build_execution_request(
            prepared_dsl=prepared_dsl,
            mechanism_source=mechanism_source,
            initial_values=values,
            execution_settings=execution_settings,
            scientific_settings=scientific_settings,
        ),
    )


def build_preview_target(
    *,
    prepared_dsl: PreparedDslRequestSet,
    key: PreviewResultKey,
    set_name: str,
) -> PreviewTarget:
    return PreviewTarget(
        key=key,
        set_name=set_name,
        prepared_dsl=prepared_dsl,
        execution_request=build_execution_request(
            prepared_dsl=prepared_dsl,
            mechanism_source=key.mechanism_source,
            initial_values=key.initial_values,
            execution_settings=key.execution_settings,
            scientific_settings=key.scientific_settings,
            parameter_values=dict(key.parameter_values),
        ),
    )


def _cancel_event(cancelled: Event):
    def event(_time: float, _state: np.ndarray) -> float:
        return -1.0 if cancelled.is_set() else 1.0

    event.terminal = True
    event.direction = -1.0
    event._kindred_cancel_event = True
    event._kindred_cancelled = cancelled.is_set
    return event


def _prepare_target(
    target: ExplicitRunTarget | PreviewTarget,
    *,
    cancelled: Event,
) -> tuple[
    PreparedSimulationRun,
    tuple[tuple[str, float, str], ...],
]:
    if cancelled.is_set():
        raise SimulationCancelled()
    request = target.execution_request
    parameters = _captured_parameter_coordinates(
        request,
        prepared_dsl=target.prepared_dsl,
    )
    prepared = prepare_simulation_worker_run(
        execution_request=request,
        events=[_cancel_event(cancelled)],
    )
    return prepared, parameters


def _captured_parameter_coordinates(
    request: SimulationExecutionRequest,
    *,
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[tuple[str, float, str], ...]:
    """Capture typed writable coordinate truth for one immutable request."""

    solver_config = dict(request.solver_config)
    cyclicity = solver_config[
        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED
    ]
    if type(cyclicity) is not bool:
        raise TypeError("Wegscheider cyclicity execution truth must be a bool.")
    coordinates = enumerate_parameter_coordinates(
        prepared_dsl,
        wegscheider_cyclicity_enabled=cyclicity,
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
    apply_parameter_algebra_spec_to_mechanism(
        prepared_dsl.parameter_algebra_spec_for_execution(),
        mechanism=mechanism,
        require_mutable=False,
    )
    return analyze_wegscheider_cyclicity(mechanism)


def _detailed_balance_correction_for_capture(
    capture: ExplicitRunCapture,
) -> DetailedBalanceCorrectionRequest | None:
    unresolved: list[
        tuple[
            MechanismAuthoringSource,
            float,
            WegscheiderCyclicityReport,
        ]
    ] = []
    reports_by_analysis_identity: dict[
        str,
        WegscheiderCyclicityReport,
    ] = {}
    for target in capture.targets:
        if not _policy_enabled(
            target.key.scientific_settings.wegscheider_policy,
            field="wegscheider_policy",
        ):
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
            unresolved.append(
                (target.key.mechanism_source, temperature, report)
            )
    if not unresolved:
        return None

    source, temperature, report = unresolved[0]
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
        or tuple(
            (
                cycle.cycle_id,
                cycle.step_indices,
                cycle.parameter_names,
            )
            for cycle in candidate_report.unresolved_cycles
        )
        != cycle_truth
        for candidate_source, candidate_temperature, candidate_report
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
            f"{source.full_dsl}\0{target_identity}\0"
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
    )


def _verify_detailed_balance_resolved(
    prepared_dsl: PreparedDslRequestSet,
) -> None:
    """Prove one accepted typed correction resolves every cycle."""

    mechanism = prepared_dsl.mechanism_for_execution()
    apply_parameter_algebra_spec_to_mechanism(
        prepared_dsl.parameter_algebra_spec_for_execution(),
        mechanism=mechanism,
        require_mutable=False,
    )
    validate_wegscheider_cyclicity_resolved(mechanism)


def _solve_prepared(
    prepared: PreparedSimulationRun,
    *,
    parameters: tuple[tuple[str, float, str], ...],
    cancelled: Event,
) -> SimulationResultValue:
    if cancelled.is_set():
        raise SimulationCancelled()
    solved = solve_ode(prepared.request)
    if cancelled.is_set():
        raise SimulationCancelled()
    finalized = evaluate_simulation_result_algebra(
        mechanism=prepared.mechanism,
        result=solved,
        species_names=list(prepared.species_names),
        initials_for_algebra=prepared.initials_for_algebra,
        simulation_plan=None,
    )
    matrix = np.asarray(finalized.y, dtype=float)
    time = np.asarray(solved.t, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(finalized.species_names):
        raise ValueError("solver returned an invalid species matrix.")
    if matrix.shape[1] != time.size:
        raise ValueError("solver result time and species lengths differ.")
    if cancelled.is_set():
        raise SimulationCancelled()
    solver_provenance = dict(getattr(solved, "provenance", {}) or {})
    segment_ids = np.asarray(
        solver_provenance.get("result_segment_ids"),
        dtype=np.int64,
    ).reshape(-1)
    segment_keys = tuple(
        str(value)
        for value in solver_provenance.get("result_segment_keys", ())
    )
    discontinuity_ids = np.asarray(
        solver_provenance.get("result_discontinuity_ids"),
        dtype=np.int64,
    ).reshape(-1)
    if not (
        segment_ids.size
        == len(segment_keys)
        == discontinuity_ids.size
        == time.size
    ):
        raise ValueError(
            "solver result is missing aligned segment/discontinuity identity."
        )
    result = SimulationResultValue(
        t=time,
        series={
            str(name): matrix[index, :]
            for index, name in enumerate(finalized.species_names)
        },
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        discontinuity_ids=discontinuity_ids,
        scalars=finalized.algebra_scalars,
        parameters=parameters,
        provenance=SimulationResultProvenance(
            solver=solver_provenance,
            preparation_warnings=tuple(prepared.warnings),
            algebra_errors=tuple(finalized.algebra_errors),
        ),
        intervention_execution=solved.intervention_execution,
    )
    if cancelled.is_set():
        raise SimulationCancelled()
    return result


class _DirectSimulationWorker(QtCore.QThread):
    target_accepted = QtCore.Signal(int, object)
    target_succeeded = QtCore.Signal(int, object, object)
    target_failed = QtCore.Signal(int, object, str, bool, bool)

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
                    outcome = _TargetExecutionOutcome(
                        key=target.key,
                        accepted=False,
                        cancelled=self._cancelled.is_set(),
                        message=str(exc) or type(exc).__name__,
                    )
                if outcome.result is not None and self._cancelled.is_set():
                    outcome = _TargetExecutionOutcome(
                        key=outcome.key,
                        accepted=True,
                        cancelled=True,
                        message="Simulation cancelled before publication.",
                    )
                if outcome.accepted:
                    self.target_accepted.emit(self._job_id, outcome.key)
                if outcome.result is not None and self._cancelled.is_set():
                    outcome = _TargetExecutionOutcome(
                        key=outcome.key,
                        accepted=True,
                        cancelled=True,
                        message="Simulation cancelled before publication.",
                    )
                if outcome.result is not None:
                    self.target_succeeded.emit(
                        self._job_id,
                        outcome.key,
                        outcome.result,
                    )
                    if self._cancelled.is_set():
                        self.target_failed.emit(
                            self._job_id,
                            outcome.key,
                            "Simulation cancelled during publication.",
                            True,
                            True,
                        )
                else:
                    self.target_failed.emit(
                        self._job_id,
                        outcome.key,
                        outcome.message,
                        outcome.cancelled,
                        outcome.accepted,
                    )
                if self._cancelled.is_set():
                    for pending in futures:
                        pending.cancel()

    def _execute_target(
        self,
        target: ExplicitRunTarget | PreviewTarget,
    ) -> _TargetExecutionOutcome:
        if self._cancelled.is_set():
            return _TargetExecutionOutcome(
                key=target.key,
                accepted=False,
                cancelled=True,
                message="Simulation cancelled before preparation.",
            )
        try:
            prepared, parameters = _prepare_target(
                target,
                cancelled=self._cancelled,
            )
        except SimulationCancelled:
            return _TargetExecutionOutcome(
                key=target.key,
                accepted=False,
                cancelled=True,
                message="Simulation cancelled.",
            )
        except BaseException as exc:  # noqa: BLE001
            return _TargetExecutionOutcome(
                key=target.key,
                accepted=False,
                cancelled=False,
                message=str(exc) or type(exc).__name__,
            )
        try:
            result = _solve_prepared(
                prepared,
                parameters=parameters,
                cancelled=self._cancelled,
            )
        except SimulationCancelled:
            return _TargetExecutionOutcome(
                key=target.key,
                accepted=True,
                cancelled=True,
                message="Simulation cancelled.",
            )
        except BaseException as exc:  # noqa: BLE001
            return _TargetExecutionOutcome(
                key=target.key,
                accepted=True,
                cancelled=False,
                message=str(exc) or type(exc).__name__,
            )
        return _TargetExecutionOutcome(
            key=target.key,
            accepted=True,
            cancelled=False,
            result=result,
        )


RepositoryResolver = Callable[[str], ResultRepository | None]
ExplicitKeyCurrentness = Callable[[ExplicitResultKey], bool]
RenderResults = Callable[[str], bool]
RawUserParallelMaximum = Callable[[], object]


class SimulationJobOwner(QtCore.QObject):
    """Owns one worker, immutable captures, and publication generations."""

    state_changed = QtCore.Signal()
    owned_qthreads_changed = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    results_changed = QtCore.Signal(str)
    terminal = QtCore.Signal(object)
    render_failed = QtCore.Signal(str, str)
    detailed_balance_correction_requested = QtCore.Signal(object)

    def __init__(
        self,
        *,
        gate: ComputationGate,
        repository_for_tab: RepositoryResolver,
        explicit_result_key_is_current: ExplicitKeyCurrentness,
        commit_detailed_balance_source: Callable[
            [str, MechanismAuthoringSource],
            PreparedDslRequestSet | None,
        ],
        render_results: RenderResults | None = None,
        user_parallel_maximum: RawUserParallelMaximum | None = None,
        cpu_count: Callable[[], object] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(gate, ComputationGate):
            raise TypeError("gate must be a ComputationGate.")
        if not callable(repository_for_tab):
            raise TypeError("repository_for_tab must be callable.")
        if not callable(explicit_result_key_is_current):
            raise TypeError("explicit_result_key_is_current must be callable.")
        if not callable(commit_detailed_balance_source):
            raise TypeError("commit_detailed_balance_source must be callable.")
        self._gate = gate
        self._repository_for_tab = repository_for_tab
        self._explicit_result_key_is_current = explicit_result_key_is_current
        self._commit_detailed_balance_source = commit_detailed_balance_source
        self._render_results = render_results
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
    ) -> None:
        tid = _required_id(tab_id, "tab_id")
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            self._current_preview_key_by_target.pop(
                (tid, sid),
                None,
            )

    def _retire_all_preview_keys_for_tab(self, tab_id: str) -> None:
        tid = _required_id(tab_id, "tab_id")
        for target in tuple(self._current_preview_key_by_target):
            if target[0] == tid:
                self._current_preview_key_by_target.pop(target, None)

    def deauthorize_targets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
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
        repository.deauthorize_sets(
            targets,
            explicit_key_is_current=self._explicit_completion_is_live_key,
        )
        self.results_changed.emit(tid)
        return generations

    def begin_preview_edit(
        self,
        identity: PreviewResultIdentity,
    ) -> PreviewResultKey | None:
        """Mint and install the next exact Preview key before debounce."""

        if not isinstance(identity, PreviewResultIdentity):
            raise TypeError("identity must be a PreviewResultIdentity.")
        if self._application_close_fenced:
            self.status_changed.emit("Preview rejected: simulation owner is closing.")
            return None
        generation = self._advance_preview_publication_generations(
            identity.tab_id,
            (identity.set_id,),
        )[0][1]
        key = PreviewResultKey.from_identity(
            identity,
            generation=generation,
        )
        repository = self._repository(identity.tab_id)
        self._drop_pending_preview_for_targets(
            identity.tab_id,
            (identity.set_id,),
        )
        self.advance_explicit_publication_generations(
            identity.tab_id,
            (identity.set_id,),
        )
        repository.capture_preview_reference(key)
        repository.authorize_preview_display(key)
        self._current_preview_key_by_target[
            (identity.tab_id, identity.set_id)
        ] = key
        self.results_changed.emit(identity.tab_id)
        return key

    def reconcile_canonical_targets(
        self,
        keys: Iterable[ExplicitResultKey],
    ) -> tuple[tuple[str, bool], ...]:
        """Semantically clear/hydrate canonical targets in one publication batch."""

        captured = tuple(keys)
        if not captured:
            return ()
        for key in captured:
            if not isinstance(key, ExplicitResultKey):
                raise TypeError("keys must contain ExplicitResultKey values.")
        tab_id = captured[0].tab_id
        if any(key.tab_id != tab_id for key in captured):
            raise ValueError("canonical-return keys must belong to one tab.")
        set_ids = tuple(key.set_id for key in captured)
        if len(set_ids) != len(set(set_ids)):
            raise ValueError("canonical-return keys must target unique sets.")

        repository = self._repository(tab_id)
        live_keys = tuple(
            key
            for key in captured
            if self._explicit_completion_is_live_key(key)
        )
        live_set_ids = tuple(key.set_id for key in live_keys)
        self._drop_pending_preview_for_targets(tab_id, live_set_ids)
        self._advance_preview_publication_generations(tab_id, live_set_ids)
        hydrated_by_set_id: dict[str, bool] = {}
        for key in live_keys:
            self._retire_preview_keys(tab_id, (key.set_id,))
            repository.clear_preview(
                key.set_id,
                clear_reference=False,
            )
            hydrated = repository.hydrate_cached(key)
            hydrated_by_set_id[key.set_id] = hydrated is not None
            if hydrated is None:
                repository.clear_preview(key.set_id)
        self.results_changed.emit(tab_id)
        return tuple(
            (key.set_id, hydrated_by_set_id.get(key.set_id, False))
            for key in captured
        )

    def purge_sets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]:
        """Purge captures/results but retain monotonic generation tombstones."""
        tid = _required_id(tab_id, "tab_id")
        targets = tuple(
            dict.fromkeys(_required_id(value, "set_id") for value in set_ids)
        )
        if not targets:
            return ()
        pending = self._pending_preview
        if pending is not None and pending.tab_id == tid and any(
            target.key.set_id in targets for target in pending.targets
        ):
            self.drop_pending_preview(tid)
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
        self._targets = tuple(
            target
            for target in self._targets
            if not (
                target.key.tab_id == tid
                and target.key.set_id in target_set
            )
        )
        generations = self.advance_explicit_publication_generations(tid, targets)
        self._repository(tid).purge_sets(targets)
        self.results_changed.emit(tid)
        return generations

    def purge_tab(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]:
        """Fence a removed tab while preserving all generation tombstones."""
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
        self._targets = tuple(
            target for target in self._targets if target.key.tab_id != tid
        )
        generations = self.advance_explicit_publication_generations(tid, targets)
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
            self.drop_pending_preview()
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
        if gate_already_owned:
            if not self._gate.is_owned_by(self):
                self.status_changed.emit(
                    "Run continuation rejected: its admission is no longer owned."
                )
                return False
        elif not self._gate.try_acquire(self, "run"):
            self.status_changed.emit("Run rejected: another computation is active.")
            return False
        self.drop_pending_preview()
        try:
            self._start_worker(
                "run",
                capture.tab_id,
                capture.targets,
                max_parallel=capacity.effective,
            )
        except Exception as exc:
            self._gate.release(self)
            self.status_changed.emit(
                f"Run dispatch failed: {type(exc).__name__}: {exc}"
            )
            return False
        self.status_changed.emit(
            f"Run started with capacity {capacity.effective}."
        )
        return True

    def complete_detailed_balance_correction(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> bool:
        """Commit, rebase, and continue one exact admitted Run once."""

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
            corrected_prepared = self._commit_detailed_balance_source(
                request.tab_id,
                corrected_source,
            )
            if not isinstance(
                corrected_prepared,
                PreparedDslRequestSet,
            ):
                raise RuntimeError(
                    "Canonical source admission rejected the correction."
                )
            _verify_detailed_balance_resolved(corrected_prepared)
            if (
                self._application_close_fenced
                or self._pending_detailed_balance is not pending
                or not self._gate.is_owned_by(self)
            ):
                raise RuntimeError(
                    "Run correction admission was retired during commit."
                )
            rebased_targets: list[ExplicitRunTarget] = []
            for target in pending.capture.targets:
                corrected_key = replace(
                    target.key,
                    mechanism_source=corrected_source,
                )
                if not self._explicit_completion_is_live_key(corrected_key):
                    raise RuntimeError(
                        f"Corrected Run target {target.set_name!r} is not "
                        "the exact live canonical target."
                    )
                rebased_targets.append(
                    ExplicitRunTarget(
                        key=corrected_key,
                        set_name=target.set_name,
                        publication_generation=(
                            self.explicit_publication_generation(
                                corrected_key.tab_id,
                                corrected_key.set_id,
                            )
                        ),
                        prepared_dsl=corrected_prepared,
                        execution_request=build_execution_request(
                            prepared_dsl=corrected_prepared,
                            mechanism_source=corrected_source,
                            initial_values=corrected_key.initial_values,
                            execution_settings=corrected_key.execution_settings,
                            scientific_settings=corrected_key.scientific_settings,
                        ),
                    )
                )
            rebased = ExplicitRunCapture(
                tab_id=pending.capture.tab_id,
                targets=tuple(rebased_targets),
            )
            self._pending_detailed_balance = None
            launched = self._launch_run_capture(
                rebased,
                gate_already_owned=True,
            )
            if not launched:
                self._gate.release(self)
                raise RuntimeError("Corrected Run could not be dispatched.")
        except Exception as exc:
            if self._pending_detailed_balance is pending:
                self._abort_pending_detailed_balance(
                    "Detailed-balance correction rejected; Run aborted: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                self.status_changed.emit(
                    "Detailed-balance correction committed but Run aborted: "
                    f"{type(exc).__name__}: {exc}"
                )
            return False
        self.status_changed.emit(
            "Detailed-balance correction committed; continuing Run once."
        )
        return True

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
        if not self._gate.try_acquire(self, "preview"):
            self._pending_preview = intent
            self.status_changed.emit(
                "Preview not admitted while another computation is active; "
                "latest intent retained."
            )
            self.state_changed.emit()
            return False
        admitted_intent = self._current_preview_intent(intent)
        if admitted_intent is None:
            self._gate.release(self)
            self.status_changed.emit(
                "Preview rejected: every exact key changed during admission."
            )
            return False
        intent = admitted_intent
        self._start_worker(
            "preview",
            intent.tab_id,
            intent.targets,
            max_parallel=1,
        )
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
        """Permanently reject every later Run and Preview admission."""

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

    def show_cached(self, key: ExplicitResultKey) -> bool:
        repository = self._repository(key.tab_id)
        if (
            not self._explicit_completion_is_live_key(key)
            or self._preview_target_is_active(key.tab_id, key.set_id)
        ):
            self.status_changed.emit(
                "Requested result is not currently display-eligible; display unchanged."
            )
            return False
        if repository.hydrate_cached(key) is None:
            self.status_changed.emit("Requested result is not cached; display unchanged.")
            return False
        self.results_changed.emit(key.tab_id)
        self.rerender(key.tab_id)
        self.status_changed.emit("Cached result shown.")
        return True

    def rerender(self, tab_id: str) -> bool:
        """One direct Try Again operation without retained recovery state."""

        callback = self._render_results
        if callback is None:
            return True
        tid = _required_id(tab_id, "tab_id")
        try:
            rendered = bool(callback(tid))
        except Exception as exc:
            self.render_failed.emit(tid, str(exc))
            return False
        if not rendered:
            self.render_failed.emit(tid, "Result surface rejected the projection.")
        return rendered

    def _start_worker(
        self,
        kind: str,
        tab_id: str,
        targets: tuple[ExplicitRunTarget | PreviewTarget, ...],
        *,
        max_parallel: int,
    ) -> None:
        self._job_id += 1
        self._job_kind = kind
        self._origin_tab_id = tab_id
        self._targets = tuple(targets)
        self._outcomes = {}
        self._details = []
        worker = _DirectSimulationWorker(
            job_id=self._job_id,
            targets=self._targets,
            max_parallel=max_parallel,
            parent=self,
        )
        worker.target_accepted.connect(self._on_target_accepted)
        worker.target_succeeded.connect(self._on_target_succeeded)
        worker.target_failed.connect(self._on_target_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self.owned_qthreads_changed.emit()
        self.state_changed.emit()
        worker.start()

    @QtCore.Slot(int, object)
    def _on_target_accepted(self, job_id: int, key: object) -> None:
        if (
            int(job_id) == self._job_id
            and self._worker is not None
            and not self._worker.cancelled
            and not self._application_close_fenced
        ):
            self._outcomes.setdefault(key, "accepted")

    @QtCore.Slot(int, object, object)
    def _on_target_succeeded(self, job_id: int, key: object, result: object) -> None:
        if (
            int(job_id) != self._job_id
            or self._worker is None
            or self._worker.cancelled
            or self._application_close_fenced
            or not isinstance(result, SimulationResultValue)
        ):
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
            repository.put_explicit(key, result)
            if self._preview_target_is_active(key.tab_id, key.set_id):
                self._advance_preview_publication_generations(
                    key.tab_id,
                    (key.set_id,),
                )
                self._retire_preview_keys(key.tab_id, (key.set_id,))
            self._outcomes[key] = "success"
            self.results_changed.emit(key.tab_id)
            self._render_after_publication(key.tab_id)
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
            repository.store_preview(key, result)
            self._outcomes[key] = "success"
            # Semantic publication is complete before any render attempt,
            # including for inactive tabs and non-plotted targets.
            self.results_changed.emit(key.tab_id)
            self._render_after_publication(key.tab_id)
            return
        self._record_discard(
            key,
            "Simulation result discarded: its job kind no longer matches.",
        )

    @QtCore.Slot(int, object, str, bool, bool)
    def _on_target_failed(
        self,
        job_id: int,
        key: object,
        message: str,
        cancelled: bool,
        accepted: bool,
    ) -> None:
        if (
            int(job_id) != self._job_id
            or self._worker is None
            or self._application_close_fenced
        ):
            return
        outcome = "cancelled" if cancelled else "failure"
        self._outcomes[key] = outcome
        target_name = self._target_display_name(key)
        stage = "execution" if accepted else "preparation"
        detail = (
            f"{target_name} {stage} "
            f"{'cancelled' if cancelled else 'failed'}: {message}"
        )
        self._details.append(detail)
        self.status_changed.emit(detail)
        # Prior current/cache entries remain untouched for preparation failure,
        # execution failure, and cancellation.

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        if self.sender() is not self._worker:
            return
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
        unresolved = tuple(
            key
            for key, value in self._outcomes.items()
            if value == "accepted"
        )
        if unresolved:
            if self._worker is not None and self._worker.cancelled:
                cancelled += len(unresolved)
                details.extend(
                    f"{self._target_display_name(key)} "
                    "cancelled before publication."
                    for key in unresolved
                )
            else:
                failed += len(unresolved)
                details.extend(
                    f"{self._target_display_name(key)} "
                    "ended without a terminal outcome."
                    for key in unresolved
                )
        unprocessed = max(0, len(self._targets) - len(self._outcomes))
        if self._worker is not None and self._worker.cancelled:
            cancelled += unprocessed
        elif unprocessed:
            failed += unprocessed
            details.append(
                f"{unprocessed} target(s) ended without preparation or execution."
            )
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
        summary = self._terminal_summary()
        self._job_id += 1
        self._finish_current()
        self.terminal.emit(summary)
        self.status_changed.emit(self._terminal_status(summary))

    def _finish_current(self) -> None:
        worker = self._worker
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

    def _record_discard(self, key: object, detail: str) -> None:
        self._outcomes[key] = "discarded"
        self._details.append(str(detail))
        self.status_changed.emit(str(detail))

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
            return bool(self._explicit_result_key_is_current(key))
        except Exception:
            return False

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

    def _render_after_publication(self, tab_id: str) -> None:
        if not self.rerender(tab_id):
            detail = (
                "Result retained, but physical rendering failed; "
                "use Try Again to rerender."
            )
            self._details.append(detail)
            self.status_changed.emit(detail)
