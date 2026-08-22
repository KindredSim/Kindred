"""
Simulation preparation utilities used by fitting and prepared/bound execution paths.

The parser-owned ``DslAnalysis`` and ``PreparedDslRequestSet`` enter here as
admitted inputs; this module coordinates typed parameter binding, solver/Jacobian
preparation, and RHS compilation for the execution paths that consume them.
"""

from __future__ import annotations

import logging
from numbers import Real
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.species import validate_concentration, validate_time_seconds
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.temperature import (
    TemperatureSchedule,
    TemperatureScheduleDictProtocol,
    TemperatureScheduleProtocol,
    coerce_temperature_schedule,
)
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    InterventionScheduleDependencyMetadata,
    InterventionScheduleError,
    intervention_schedule_parameter_dependencies,
    intervention_schedule_parameter_names,
)
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
    compile_intervention_schedule,
)
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.solver_types import (
    SimulationRequest,
    SimulationCancellationControl,
)
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
)
from kindred.core.simulator.dsl_parse import (
    DslAnalysis,
    PreparedDslRequestSet,
)
from kindred.core.simulation_preparation_stages import (
    PreparedSolverConfig as _PreparedSolverConfig,
    build_prepared_run_context,
    build_solver_config,
)
from kindred.core.time_grid import build_time_grid

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kindred.core.mechanism_source import MechanismAuthoringSource
    from kindred.core.simulator.wegscheider_symbolic import (
        WegscheiderCyclicityReport,
        WegscheiderExecutableSatisfactionReport,
    )


def _freeze_prepared_value(value: Any) -> Any:
    """Freeze one retained preparation value without owning its policy."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_prepared_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_prepared_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_prepared_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


def _mutable_prepared_value(value: Any) -> Any:
    """Project a retained value back to ordinary serializable containers."""

    if isinstance(value, Mapping):
        return {
            str(key): _mutable_prepared_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_mutable_prepared_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return [_mutable_prepared_value(item) for item in value]
    return value

__all__ = [
    "SimulationExecutionRequest",
    "PreparedSimulationRun",
    "SimulationPreparationError",
    "build_simulation_request_from_prepared_run",
    "metadata_view_for_mechanism",
    "bind_simulation_execution_request_interventions",
    "build_simulation_execution_request",
    "solver_config_for_execution_settings",
    "prepare_simulation_worker_run",
    "resolve_prepared_run_intervention_execution",
    "fit_simulation_error_from_preparation_error",
    "prepare_preparation_failure",
    "raise_unowned_request_parameter_values",
    "reject_requested_algebra_owned_mechanism_parameters_for_fitting",
    "scalar_parameter_is_derived",
    "simulation_execution_requests_equal",
]


def _simulation_container_values_equal(left: object, right: object) -> bool:
    """Compare only the plain containers used by retained request values."""

    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
    if type(left) is ParameterAlgebraSpec:
        return _parameter_algebra_specs_equal(left, right)
    if isinstance(left, TemperatureSchedule):
        return _temperature_schedules_equal(left, right)
    if isinstance(left, Mapping):
        other = right
        if len(left) != len(other):  # type: ignore[arg-type]
            return False
        return all(
            key in other  # type: ignore[operator]
            and _simulation_container_values_equal(value, other[key])  # type: ignore[index]
            for key, value in left.items()
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _simulation_container_values_equal(item, other)
            for item, other in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _simulation_container_values_equal(item, other)
            for item, other in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    if isinstance(left, (set, frozenset)):
        return left == right
    if isinstance(left, (str, bytes, int, float, bool, type(None))):
        return bool(left == right)
    return False


def _temperature_schedules_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if type(left) is not type(right):
        return False
    if isinstance(left, TemperatureSchedule):
        return _simulation_container_values_equal(left.to_dict(), right.to_dict())  # type: ignore[union-attr]
    if isinstance(left, TemperatureScheduleDictProtocol):
        return _simulation_container_values_equal(left.to_dict(), right.to_dict())
    if isinstance(left, Mapping):
        return _simulation_container_values_equal(left, right)
    return False


def _intervention_executions_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if type(left) is not InterventionExecutionContract or type(right) is not InterventionExecutionContract:
        return False
    return (
        _simulation_container_values_equal(left.declarative_payload, right.declarative_payload)
        and _simulation_container_values_equal(left.executable_payload, right.executable_payload)
        and left.instants == right.instants
        and left.fluxes == right.fluxes
        and left.holds == right.holds
        and left.triggers == right.triggers
        and left.discontinuities == right.discontinuities
        and left.segment_identity == right.segment_identity
        and _simulation_container_values_equal(left.protocol_lineage, right.protocol_lineage)
        and left.trigger_firings == right.trigger_firings
    )


def _parameter_algebra_specs_equal(left: object, right: object) -> bool:
    if type(left) is not ParameterAlgebraSpec or type(right) is not ParameterAlgebraSpec:
        return False
    return (
        left.param_statements == right.param_statements
        and left.observable_names == right.observable_names
        and left.mechanism_namespace == right.mechanism_namespace
        and left.scalar_input_names == right.scalar_input_names
        and left.parameter_block == right.parameter_block
        and left.observable_block == right.observable_block
        and _simulation_container_values_equal(
            left.raw_to_canonical_identifiers_by_assignment,
            right.raw_to_canonical_identifiers_by_assignment,
        )
    )


def simulation_execution_requests_equal(
    left: SimulationExecutionRequest,
    right: SimulationExecutionRequest,
) -> bool:
    """Compare one accepted owner identity and its typed target values."""

    if type(left) is not SimulationExecutionRequest or type(right) is not SimulationExecutionRequest:
        return False
    return (
        left.prepared_dsl is right.prepared_dsl
        and left.species_names == right.species_names
        and left.y0 == right.y0
        and _temperature_schedules_equal(left.temperature_schedule, right.temperature_schedule)
        and _parameter_algebra_specs_equal(left.parameter_algebra_spec, right.parameter_algebra_spec)
        and _intervention_executions_equal(left.intervention_execution, right.intervention_execution)
        and left.t_span == right.t_span
        and type(left.solver_config) is type(right.solver_config)
        and left.solver_config.solver_input == right.solver_config.solver_input
        and left.solver_config.solver == right.solver_config.solver
        and left.solver_config.rtol == right.solver_config.rtol
        and left.solver_config.atol == right.solver_config.atol
        and _simulation_container_values_equal(left.solver_config.grid, right.solver_config.grid)
        and left.solver_config.use_analytic_jacobian == right.solver_config.use_analytic_jacobian
        and left.solver_config.wegscheider_cyclicity_enabled == right.solver_config.wegscheider_cyclicity_enabled
        and left.wegscheider_admission_report == right.wegscheider_admission_report
        and left.parameterized_intervention_schedule == right.parameterized_intervention_schedule
        and left.requested_parameter_names == right.requested_parameter_names
        and left.candidate_dependent_intervention_parameter_names == right.candidate_dependent_intervention_parameter_names
        and _simulation_container_values_equal(left.parameter_overrides, right.parameter_overrides)
        and left.mechanism_source == right.mechanism_source
    )


def _parameter_override_target_kind(mechanism: Any, name: str) -> str | None:
    from kindred.core.simulator.step_indexing import lookup_step_param_target

    target_name = _canonical_step_override_name(mechanism, name)
    if lookup_step_param_target(mechanism, target_name) is not None:
        return "mechanism" if _prepared_parameter_override_can_apply(mechanism, name) else "unbound_mechanism"
    if _scalar_parameter_override_known(mechanism, name):
        return "scalar"
    return None


def _classify_request_parameter_names(
    *,
    mechanism: Any,
    parameter_overrides: Mapping[str, Any] | None,
    unresolved_intervention_schedule: InterventionSchedule | None,
    raw_values: dict[str, float],
    mechanism_names: set[str],
    unbound_mechanism_names: set[str],
    schedule_names: set[str],
    scalar_names: set[str],
    unknown_names: set[str],
    mechanism_name_by_raw: dict[str, str],
    unbound_mechanism_name_by_raw: dict[str, str],
    schedule_name_by_raw: dict[str, str],
    invalid_messages: dict[str, str],
    requested_parameter_names: Iterable[str] | None = None,
    declared_scalar_names: Iterable[str] | None = None,
    validate_values: bool = True,
) -> None:
    raw_values.update(
        {
            str(name): float(value)
            for name, value in _coerce_parameter_override_items(
                parameter_overrides,
                require_finite=bool(validate_values),
            )
        }
    )
    classification_names = list(raw_values)
    for requested_name in requested_parameter_names or ():
        name = str(requested_name or "").strip()
        if name and name not in classification_names:
            classification_names.append(name)
    schedule_names.update(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    for schedule_name in schedule_names:
        name = str(schedule_name or "").strip()
        if name and name not in classification_names:
            classification_names.append(name)
    if not raw_values and not classification_names:
        return
    declared_scalar_names = {
        str(name or "").strip()
        for name in declared_scalar_names or ()
        if str(name or "").strip()
    }
    mechanism_namespace = build_namespace_from_mechanism(mechanism)
    for name in classification_names:
        claimed = False
        resolution = mechanism_namespace.resolve(name)
        if resolution.canonical_name is not None:
            canonical_name = str(resolution.canonical_name)
            target_kind = _parameter_override_target_kind(mechanism, canonical_name)
            if target_kind == "mechanism":
                mechanism_names.add(canonical_name)
                mechanism_name_by_raw[name] = canonical_name
                if name in schedule_names:
                    schedule_name_by_raw[name] = canonical_name
                claimed = True
            elif target_kind == "unbound_mechanism":
                unbound_mechanism_names.add(canonical_name)
                unbound_mechanism_name_by_raw[name] = canonical_name
                if name in schedule_names:
                    schedule_name_by_raw[name] = canonical_name
                claimed = True
            if claimed:
                continue
        invalid_message = invalid_request_parameter_identifier_message(mechanism, name)
        if invalid_message is not None:
            invalid_messages[name] = invalid_message
            continue
        if name in raw_values and scalar_parameter_is_derived(mechanism, name):
            invalid_messages[name] = (
                f"derived scalar parameter {name!r} cannot be used as direct runtime "
                "or schedule override authority."
            )
            continue
        if name in schedule_names:
            schedule_name_by_raw[name] = name
            claimed = True
        if name in declared_scalar_names:
            scalar_names.add(name)
            claimed = True
        target_kind = _parameter_override_target_kind(mechanism, name)
        if target_kind == "mechanism":
            canonical_name = _canonical_step_override_name(mechanism, name)
            mechanism_names.add(canonical_name)
            mechanism_name_by_raw[name] = canonical_name
            claimed = True
        elif target_kind == "unbound_mechanism":
            canonical_name = _canonical_step_override_name(mechanism, name)
            unbound_mechanism_names.add(canonical_name)
            unbound_mechanism_name_by_raw[name] = canonical_name
            claimed = True
        elif target_kind == "scalar":
            scalar_names.add(name)
            claimed = True
        if not claimed:
            unknown_names.add(name)

def raise_unowned_request_parameter_values(
    *,
    raw_values: Mapping[str, float],
    mechanism_name_by_raw: Mapping[str, str],
    unbound_mechanism_name_by_raw: Mapping[str, str],
    unbound_mechanism_names: Iterable[str],
    unknown_names: Iterable[str],
    invalid_messages: Mapping[str, str],
    allow_unbound_mechanism_parameters: bool = False,
) -> None:
    if invalid_messages:
        first_name = sorted(invalid_messages)[0]
        raise ValueError(invalid_messages[first_name])
    owner_to_raw: dict[str, str] = {}
    for raw_name, owner_name in sorted(
        {
            **mechanism_name_by_raw,
            **unbound_mechanism_name_by_raw,
        }.items()
    ):
        if raw_name not in raw_values:
            continue
        owner = str(owner_name)
        previous = owner_to_raw.get(owner)
        if previous is not None and previous != raw_name:
            raise ValueError(
                f"Request parameter names {previous!r} and {raw_name!r} both resolve to {owner!r}."
            )
        owner_to_raw[owner] = str(raw_name)
    unknown = sorted(str(name) for name in unknown_names)
    if unknown:
        raise ValueError(
            "Unknown request parameter(s): "
            + ", ".join(repr(name) for name in unknown)
            + ". Request parameters must belong to the mechanism, schedule, scalar algebra, or initial-condition namespace."
        )
    unbound = frozenset(str(name) for name in unbound_mechanism_names)
    if unbound and not allow_unbound_mechanism_parameters:
        first_name = sorted(unbound)[0]
        raw_names = sorted(
            raw_name
            for raw_name, canonical_name in unbound_mechanism_name_by_raw.items()
            if canonical_name == first_name
        )
        raw_clause = f" from request name {raw_names[0]!r}" if raw_names else ""
        raise ValueError(
            f"Request parameter{raw_clause} resolves to mechanism parameter {first_name!r}, "
            "but no prepared execution binding can consume it."
        )
class SimulationPreparationError(RuntimeError):
    """
    Error raised when preparing a SimulationWorker run payload/request fails.

    The GUI worker maps `stage` values to user-facing messages while keeping the
    preparation logic in core.
    """

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        parameter_names: Iterable[str] = (),
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.stage = str(stage or "unknown")
        self.parameter_names = tuple(
            sorted(
                {
                    str(name).strip()
                    for name in parameter_names
                    if str(name).strip()
                }
            )
        )
        normalized_details = dict(details or {})
        normalized_details.update(
            {
                "stage": self.stage,
                "parameter_names": self.parameter_names,
            }
        )
        self.details = MappingProxyType(
            {
                str(key): _freeze_prepared_value(value)
                for key, value in normalized_details.items()
            }
        )
        self.message = str(message or "")
        super().__init__(self.message)


def _raise_if_cancelled(
    cancellation_control: SimulationCancellationControl | None,
) -> None:
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise SimulationCancelled()


@dataclass(frozen=True)
class PreparedSimulationRun:
    """Execution-ready simulation payload produced by core preparation utilities."""

    mechanism: Any
    rhs: Callable[..., np.ndarray]
    y0: np.ndarray
    species_names: tuple[str, ...]
    temperature_schedule: TemperatureScheduleProtocol | None
    parameter_algebra_spec: Any
    intervention_execution: InterventionExecutionContract | None
    symbolic_jacobian: SymbolicJacobianExecution
    initials_for_algebra: Optional[Mapping[str, float]]
    warnings: tuple[str, ...]
    solver_config: _PreparedSolverConfig
    request: SimulationRequest | SimulationExecutionRequest
    unresolved_intervention_schedule: InterventionSchedule | None = None
    cancellation_control: SimulationCancellationControl | None = None
    wegscheider_admission_report: WegscheiderCyclicityReport | None = None
    wegscheider_executable_report: WegscheiderExecutableSatisfactionReport | None = None
    version: int = 1
    mechanism_text_len: int = 0
    param_names: tuple[str, ...] = ()
    t_end: float = 0.0
    num_points: int = 0
    temperature_K: float = 0.0
    initial_prefix: str = "init:"
    symbolic_jacobian_status: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        from kindred.core.simulator.wegscheider_symbolic import (
            WegscheiderCyclicityReport,
            WegscheiderExecutableSatisfactionReport,
        )

        y0 = np.asarray(self.y0, dtype=float).reshape(-1).copy()
        y0.setflags(write=False)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "species_names", tuple(self.species_names))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        object.__setattr__(self, "param_names", tuple(str(name) for name in self.param_names if str(name).strip()))
        object.__setattr__(self, "initial_prefix", str(self.initial_prefix or "init:"))
        if not isinstance(self.solver_config, _PreparedSolverConfig):
            raise TypeError("solver_config must be PreparedSolverConfig.")
        if isinstance(self.request, SimulationExecutionRequest):
            if self.request.solver_config != self.solver_config:
                raise ValueError(
                    "Prepared run request and solver policy must share one authority."
                )
        elif isinstance(self.request, SimulationRequest):
            if (
                self.request.solver != self.solver_config.solver
                or self.request.rtol != self.solver_config.rtol
                or self.request.atol != self.solver_config.atol
                or self.request.grid != self.solver_config.grid
            ):
                raise ValueError(
                    "Prepared solver request conflicts with its typed solver policy."
                )
        else:
            raise TypeError("Prepared run request has an unsupported type.")
        if self.wegscheider_admission_report is not None and not isinstance(
            self.wegscheider_admission_report,
            WegscheiderCyclicityReport,
        ):
            raise TypeError(
                "prepared run Wegscheider admission must be a typed report."
            )
        if self.wegscheider_executable_report is not None and not isinstance(
            self.wegscheider_executable_report,
            WegscheiderExecutableSatisfactionReport,
        ):
            raise TypeError(
                "prepared run Wegscheider executable satisfaction must be a typed report."
            )
        if self.initials_for_algebra is not None:
            if not isinstance(self.initials_for_algebra, Mapping):
                raise TypeError("initials_for_algebra must be a mapping or None.")
            object.__setattr__(
                self,
                "initials_for_algebra",
                _freeze_prepared_value(self.initials_for_algebra),
            )
        if self.symbolic_jacobian_status is not None:
            if not isinstance(self.symbolic_jacobian_status, Mapping):
                raise TypeError("symbolic_jacobian_status must be a mapping or None.")
            object.__setattr__(self, "symbolic_jacobian_status", _freeze_prepared_value(self.symbolic_jacobian_status))

    def to_serializable_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": int(self.version),
            "mechanism_text_len": int(self.mechanism_text_len),
            "param_names": [str(name) for name in self.param_names],
            "t_end": float(self.t_end),
            "num_points": int(self.num_points),
            "temperature_K": float(self.temperature_K),
            "solver_requested": self.solver_config.solver_input,
            "solver_normalized": self.solver_config.solver,
            "rtol": self.solver_config.rtol,
            "atol": self.solver_config.atol,
            "use_analytic_jacobian": self.solver_config.use_analytic_jacobian,
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: self.solver_config.wegscheider_cyclicity_enabled,
            "initial_prefix": str(self.initial_prefix),
        }
        if self.symbolic_jacobian_status:
            payload["symbolic_jacobian_status"] = _mutable_prepared_value(self.symbolic_jacobian_status)
        return payload


def resolve_prepared_run_intervention_execution(
    prepared_run: PreparedSimulationRun,
    *,
    parameter_values: Mapping[str, float],
    cancellation_check: Callable[[], bool] | None = None,
) -> InterventionExecutionContract:
    """Return the one resolved Run-owned contract consumed by Fit."""

    if cancellation_check is not None and cancellation_check():
        raise FittingCancelled()

    schedule = prepared_run.unresolved_intervention_schedule
    if schedule is None or not schedule.is_parameterized:
        return prepared_run.intervention_execution
    if cancellation_check is not None and cancellation_check():
        raise FittingCancelled()
    request = prepared_run.request
    if not isinstance(request, SimulationExecutionRequest):
        raise RuntimeError(
            "A declarative prepared schedule requires its typed execution request."
        )
    dependency_metadata = _expanded_intervention_schedule_dependency_metadata(
        schedule,
        mechanism=prepared_run.mechanism,
        parameter_algebra_spec=request.parameter_algebra_spec,
    )
    try:
        resolved_schedule = schedule.resolve_parameters(
            parameter_values,
            dependency_metadata=dependency_metadata,
        )
    except InterventionScheduleError as exc:
        details = dict(getattr(exc, "details", {}) or {})
        dependencies = tuple(
            getattr(exc, "value_dependencies", ())
            or details.get("value_dependencies", ())
        )
        if dependencies:
            details["value_dependencies"] = dependencies
        raise SimulationPreparationError(
            "intervention_schedule",
            str(exc),
            parameter_names=dependencies,
            details=details,
        ) from exc
    try:
        return compile_intervention_schedule(
            resolved_schedule,
            cancellation_check=cancellation_check,
            dependency_metadata=dependency_metadata,
        )
    except InterventionScheduleError as exc:
        details = dict(getattr(exc, "details", {}) or {})
        dependencies = tuple(
            getattr(exc, "value_dependencies", ())
            or details.get("value_dependencies", ())
        )
        if dependencies:
            details["value_dependencies"] = dependencies
        raise SimulationPreparationError(
            "intervention_schedule",
            str(exc),
            parameter_names=dependencies,
            details=details,
        ) from exc


def build_simulation_request_from_prepared_run(
    prepared_run: PreparedSimulationRun,
    *,
    y0: np.ndarray,
    intervention_execution: InterventionExecutionContract,
    symbolic_jacobian: SymbolicJacobianExecution,
    cancellation_control: SimulationCancellationControl | None = None,
) -> SimulationRequest:
    request = prepared_run.request
    cancellation_control = (
        cancellation_control
        if cancellation_control is not None
        else prepared_run.cancellation_control
    )
    _raise_if_cancelled(cancellation_control)
    if isinstance(request, SimulationExecutionRequest):
        prepared_solver_config = prepared_run.solver_config
        return SimulationRequest(
            rhs=prepared_run.rhs,
            t_span=request.t_span,
            y0=np.asarray(y0, dtype=float).reshape(-1),
            intervention_execution=intervention_execution,
            symbolic_jacobian=symbolic_jacobian,
            solver=prepared_solver_config.solver,
            rtol=prepared_solver_config.rtol,
            atol=prepared_solver_config.atol,
            grid=prepared_solver_config.grid,
            temperature_schedule=prepared_run.temperature_schedule,
            species_names=tuple(prepared_run.species_names),
            cancellation_control=cancellation_control,
        )
    if not isinstance(request, SimulationRequest):
        raise SimulationPreparationError(
            "simulation_request",
            "Prepared run request has an unsupported type.",
        )
    result = SimulationRequest(
        rhs=request.rhs,
        t_span=tuple(map(float, request.t_span)),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        intervention_execution=intervention_execution,
        symbolic_jacobian=symbolic_jacobian,
        solver=prepared_run.solver_config.solver,
        rtol=prepared_run.solver_config.rtol,
        atol=prepared_run.solver_config.atol,
        grid=prepared_run.solver_config.grid,
        temperature_schedule=request.temperature_schedule,
        species_names=tuple(prepared_run.species_names),
        cancellation_control=cancellation_control,
    )
    _raise_if_cancelled(cancellation_control)
    return result


@dataclass(frozen=True, slots=True)
class SimulationExecutionRequest:
    """Exact direct request for one target of an accepted prepared owner."""

    analysis: DslAnalysis
    prepared_dsl: PreparedDslRequestSet
    species_names: tuple[str, ...]
    y0: tuple[float, ...]
    temperature_schedule: TemperatureScheduleProtocol | None
    parameter_algebra_spec: Any
    intervention_execution: InterventionExecutionContract | None
    t_span: tuple[float, float]
    solver_config: _PreparedSolverConfig
    wegscheider_admission_report: WegscheiderCyclicityReport | None = None
    parameterized_intervention_schedule: InterventionSchedule | None = None
    requested_parameter_names: tuple[str, ...] = ()
    candidate_dependent_intervention_parameter_names: frozenset[str] = frozenset()
    parameter_overrides: Mapping[str, float] | None = None
    mechanism_source: MechanismAuthoringSource | None = None

    def __post_init__(self) -> None:
        from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport

        if type(self.analysis) is not DslAnalysis:
            raise TypeError("execution request analysis must be DslAnalysis.")
        if type(self.prepared_dsl) is not PreparedDslRequestSet:
            raise TypeError("execution request prepared_dsl must be PreparedDslRequestSet.")
        if self.analysis != self.prepared_dsl.analysis:
            raise ValueError("execution request analysis must be the accepted owner analysis.")
        species_names = tuple(self.species_names)
        if (
            not species_names
            or any(
                type(name) is not str
                or not name
                or name != name.strip()
                for name in species_names
            )
            or len(species_names) != len(set(species_names))
        ):
            raise ValueError(
                "execution request species_names must be unique nonblank strings."
            )
        try:
            y0 = tuple(
                validate_concentration(value, field="execution request initial concentration")
                for value in self.y0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "execution request y0 must contain one admitted concentration value per species."
            ) from exc
        if len(y0) != len(species_names):
            raise ValueError("execution request y0 must contain one value per species.")
        if self.temperature_schedule is not None:
            coerce_temperature_schedule(self.temperature_schedule)
        if self.parameter_algebra_spec is None:
            raise TypeError("execution request requires its retained parameter algebra spec.")
        if self.wegscheider_admission_report is not None and not isinstance(
            self.wegscheider_admission_report, WegscheiderCyclicityReport
        ):
            raise TypeError("execution request Wegscheider admission must be a typed report.")
        schedule = self.parameterized_intervention_schedule
        contract = self.intervention_execution
        if schedule is not None:
            if type(schedule) is not InterventionSchedule or not schedule.is_parameterized:
                raise TypeError(
                    "execution request declarative intervention state must be an exact parameterized schedule."
                )
            if contract is not None:
                raise ValueError(
                    "execution request intervention state cannot be both declarative and compiled."
                )
        elif not isinstance(contract, InterventionExecutionContract):
            raise TypeError("bound execution requests require one compiled intervention contract.")
        requested_names = tuple(str(name).strip() for name in self.requested_parameter_names)
        if any(not name for name in requested_names) or len(requested_names) != len(set(requested_names)):
            raise ValueError(
                "execution request requested parameter names must be unique and nonblank."
            )
        candidate_dependent_names = frozenset(
            str(name).strip()
            for name in self.candidate_dependent_intervention_parameter_names
            if str(name).strip()
        )
        if candidate_dependent_names and schedule is None:
            raise ValueError(
                "candidate-dependent intervention parameters require a declarative schedule."
            )
        if type(self.t_span) is not tuple or len(self.t_span) != 2:
            raise TypeError("execution request t_span must be an exact pair.")
        try:
            t_span = (
                validate_time_seconds(self.t_span[0], field="execution t_start"),
                validate_time_seconds(self.t_span[1], field="execution t_end"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "execution request requires admitted 0 <= t_start < t_end."
            ) from exc
        if t_span[1] <= t_span[0]:
            raise ValueError("execution request requires t_end > t_start.")
        if not isinstance(self.solver_config, _PreparedSolverConfig):
            raise TypeError(
                "execution request solver_config must be PreparedSolverConfig."
            )
        if self.parameter_overrides is not None and not isinstance(
            self.parameter_overrides,
            Mapping,
        ):
            raise TypeError(
                "execution request parameter_overrides must be a mapping."
            )
        overrides = (
            None
            if self.parameter_overrides is None
            else {
                str(name): float(value)
                for name, value in self.parameter_overrides.items()
            }
        )
        if overrides is not None and any(
            not name.strip() or not np.isfinite(value)
            for name, value in overrides.items()
        ):
            raise ValueError(
                "execution request parameter overrides must be finite and "
                "have nonblank names."
            )
        if self.mechanism_source is not None:
            from kindred.core.mechanism_source import (
                MechanismAuthoringSource,
                source_matches_analysis,
            )

            if not isinstance(self.mechanism_source, MechanismAuthoringSource):
                raise TypeError(
                    "execution request mechanism_source must be typed authoring "
                    "source when present."
                )
            if not source_matches_analysis(
                self.mechanism_source,
                self.analysis,
            ):
                raise ValueError(
                    "execution request mechanism source does not match its "
                    "admitted analysis."
                )
        object.__setattr__(self, "species_names", species_names)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "requested_parameter_names", requested_names)
        object.__setattr__(self, "candidate_dependent_intervention_parameter_names", candidate_dependent_names)
        object.__setattr__(self, "t_span", t_span)
        object.__setattr__(self, "parameter_overrides", overrides)

_MISSING = object()


def _fresh_preparation_unresolved_intervention_schedule_for_request(
    execution_request: SimulationExecutionRequest,
) -> InterventionSchedule | None:
    return execution_request.parameterized_intervention_schedule


def _parameter_algebra_dependency_graph(
    mechanism: Any,
    parameter_algebra_spec: Any,
) -> Dict[str, frozenset[str]]:
    """Return terminal solver-coordinate dependencies for retained Algebra.

    Intervention schedules are compiled once when their inputs are literal or
    fixed.  A schedule must remain declarative when any requested Fit
    coordinate can change one of its values, including through a chain of
    retained ``param`` assignments.  The retained Algebra evaluation model is
    the sole direct-dependency authority; this helper resolves that graph to
    terminal causal leaves and never evaluates or rewrites science.
    """

    assignments = tuple(getattr(parameter_algebra_spec, "param_statements", ()) or ())
    if not assignments:
        return {}

    from kindred.core.simulator.parameter_algebra import read_mechanism_parameter_values
    from kindred.core.simulator.parameter_algebra_eval import (
        build_parameter_algebra_evaluation_model,
    )

    base_values = read_mechanism_parameter_values(
        mechanism,
        names=set(getattr(parameter_algebra_spec, "mechanism_param_names", ()) or ()),
    )
    scalar_values = _scalar_schedule_resolution_values_from_mechanism(mechanism)
    base_values.update(scalar_values)
    model = build_parameter_algebra_evaluation_model(
        parameter_algebra_spec,
        assignments=assignments,
        base_values=base_values,
    )
    assignment_names = {
        str(getattr(assignment, "name", "") or "").strip()
        for assignment in assignments
    }
    if "" in assignment_names:
        raise prepare_preparation_failure(
            "parameter_algebra",
            "Parameter algebra dependency graph contains an assignment with no name.",
        )

    direct: dict[str, frozenset[str]] = {}
    for assignment in assignments:
        name = str(getattr(assignment, "name", "") or "").strip()
        solver_dependencies = {
            str(dependency).strip()
            for dependency in model.solver_dependencies_by_assignment.get(name, ())
            if str(dependency).strip()
        }
        assignment_dependencies = {
            str(dependency).strip()
            for dependency in model.assignment_dependencies_by_assignment.get(name, ())
            if str(dependency).strip()
        }
        dependencies = frozenset(solver_dependencies | assignment_dependencies)
        if name in dependencies:
            raise prepare_preparation_failure(
                "parameter_algebra",
                f"Parameter algebra dependency graph contains a self-reference for {name!r}.",
            )
        direct[name] = dependencies

    namespace = build_namespace_from_mechanism(mechanism)

    def canonical_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        if not name:
            raise prepare_preparation_failure(
                "parameter_algebra",
                "Parameter algebra dependency graph contains an unnamed dependency.",
            )
        resolution = namespace.resolve(name)
        return str(resolution.canonical_name or name)

    terminal: dict[str, frozenset[str]] = {}

    def visit(name: str, active: tuple[str, ...]) -> frozenset[str]:
        cached = terminal.get(name)
        if cached is not None:
            return cached
        if name in active:
            cycle = (*active[active.index(name) :], name)
            raise prepare_preparation_failure(
                "parameter_algebra",
                "Parameter algebra dependency graph contains a dependency cycle: "
                + " -> ".join(cycle),
            )

        dependencies = direct[name]
        if not dependencies:
            result = frozenset({canonical_name(name)})
            terminal[name] = result
            return result

        leaves: set[str] = set()
        next_active = (*active, name)
        for dependency in sorted(dependencies):
            if dependency == name:
                raise prepare_preparation_failure(
                    "parameter_algebra",
                    f"Parameter algebra dependency graph contains a self-reference for {name!r}.",
                )
            if dependency in direct:
                leaves.update(visit(dependency, next_active))
            else:
                leaves.add(canonical_name(dependency))
        if not leaves:
            raise prepare_preparation_failure(
                "parameter_algebra",
                f"Parameter algebra dependency graph produced no terminal dependency for {name!r}.",
            )
        result = frozenset(leaves)
        terminal[name] = result
        return result

    for name in direct:
        visit(name, ())
    return terminal


def _expanded_intervention_schedule_dependency_metadata(
    schedule: InterventionSchedule | None,
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
) -> InterventionScheduleDependencyMetadata:
    """Expand authored schedule references into final optimizer coordinates."""
    direct_metadata = intervention_schedule_parameter_dependencies(schedule)
    if not direct_metadata:
        return direct_metadata
    graph = _parameter_algebra_dependency_graph(mechanism, parameter_algebra_spec)
    namespace = build_namespace_from_mechanism(mechanism)

    def canonical_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        resolution = namespace.resolve(name)
        return str(resolution.canonical_name or name)

    expanded: dict[tuple[int, int | None, str], tuple[str, ...]] = {}
    for identity, raw_names in direct_metadata.items():
        dependencies: set[str] = set()
        for raw_name in raw_names:
            root = canonical_name(raw_name)
            terminal_leaves = graph.get(root)
            if terminal_leaves is None:
                # Non-Algebra schedule roots remain direct dependencies.
                dependencies.add(root)
                continue
            # The graph already contains only terminal mechanism/scalar leaves;
            # do not re-expand or filter implementation assignments here.
            dependencies.update(terminal_leaves)
        expanded[identity] = tuple(sorted(dependencies))
    return MappingProxyType(expanded)


def _resolve_intervention_schedule_for_request(
    schedule: InterventionSchedule | None,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    dependency_metadata: InterventionScheduleDependencyMetadata | None = None,
) -> InterventionSchedule | None:
    if schedule is None:
        return None
    if parameter_values is None and schedule.is_parameterized:
        raise SimulationPreparationError(
            "intervention_schedule",
            "Parameterized intervention schedules require typed request parameter classification before resolution.",
        )
    values = dict(parameter_values or {})
    try:
        return schedule.resolve_parameters(
            values,
            dependency_metadata=dependency_metadata,
        )
    except InterventionScheduleError as exc:
        details = dict(getattr(exc, "details", {}) or {})
        dependencies = tuple(
            getattr(exc, "value_dependencies", ())
            or details.get("value_dependencies", ())
        )
        if dependencies:
            details["value_dependencies"] = dependencies
        raise SimulationPreparationError(
            "intervention_schedule",
            str(exc),
            parameter_names=dependencies,
            details=details,
        ) from exc


def _schedule_parameters_deferred_to_prepared_request(
    execution_request: SimulationExecutionRequest,
    unresolved_intervention_schedule: InterventionSchedule,
) -> frozenset[str]:
    if not unresolved_intervention_schedule.is_parameterized:
        return frozenset()
    retained = execution_request.candidate_dependent_intervention_parameter_names
    if retained:
        return retained
    requested = execution_request.requested_parameter_names
    supplied = set(dict(execution_request.parameter_overrides or {}).keys())
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    if not requested or not referenced:
        return frozenset()

    mechanism = execution_request.prepared_dsl.accepted_mechanism_for_structure()
    spec = execution_request.parameter_algebra_spec
    namespace = build_namespace_from_mechanism(mechanism)

    def canonical_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        resolution = namespace.resolve(name)
        return str(resolution.canonical_name or name)

    dependency_graph = _parameter_algebra_dependency_graph(
        mechanism,
        spec,
    )
    requested_names = {
        canonical_name(name): str(name)
        for name in requested
        if str(name).strip()
    }
    supplied_names = {canonical_name(name) for name in supplied if str(name).strip()}
    deferred: set[str] = set()
    for raw_schedule_name in referenced:
        schedule_name = canonical_name(raw_schedule_name)
        schedule_dependencies = set(dependency_graph.get(schedule_name, ()))
        schedule_dependencies.add(schedule_name)
        for candidate_name in requested_names:
            if candidate_name in schedule_dependencies and candidate_name not in supplied_names:
                deferred.add(str(raw_schedule_name))
    return frozenset(deferred)


def bind_simulation_execution_request_interventions(
    request: SimulationExecutionRequest,
) -> SimulationExecutionRequest:
    """Retain declarative target schedules for worker-local resolution."""

    if not isinstance(request, SimulationExecutionRequest):
        raise TypeError(
            "intervention binding requires SimulationExecutionRequest."
        )
    unresolved = request.parameterized_intervention_schedule
    if unresolved is None:
        return request
    deferred = _schedule_parameters_deferred_to_prepared_request(request, unresolved)
    if deferred:
        return replace(
            request,
            candidate_dependent_intervention_parameter_names=deferred,
        )
    return request


def solver_config_for_execution_settings(
    *,
    execution_settings: Any,
    scientific_settings: Any,
) -> _PreparedSolverConfig:
    """Project accepted tab settings into the sole direct-request policy."""

    from kindred.core.project_document import (
        SimulationExecutionSettings,
        TabScientificSettings,
        validate_simulation_execution_settings,
        validate_tab_scientific_settings,
    )

    if not isinstance(execution_settings, SimulationExecutionSettings):
        raise TypeError(
            "execution_settings must be a SimulationExecutionSettings value."
        )
    if not isinstance(scientific_settings, TabScientificSettings):
        raise TypeError("scientific_settings must be a TabScientificSettings value.")
    validate_simulation_execution_settings(execution_settings)
    validate_tab_scientific_settings(scientific_settings)
    return build_solver_config(
        solver_input=execution_settings.solver,
        rtol=scientific_settings.rtol,
        atol=scientific_settings.atol,
        grid={"N": execution_settings.integration_points},
        use_analytic_jacobian=(
            scientific_settings.analytic_jacobian_policy
        ),
        wegscheider_cyclicity_enabled=(
            scientific_settings.wegscheider_policy
        ),
    )


def build_simulation_execution_request(
    *,
    prepared_dsl: PreparedDslRequestSet,
    mechanism_source: Any,
    initial_values: Iterable[tuple[str, float]],
    execution_settings: Any,
    scientific_settings: Any,
    parameter_values: Mapping[str, float] | None = None,
) -> SimulationExecutionRequest:
    """Capture, isolate, bind, and compile one direct Run/Preview target."""

    from kindred.core.mechanism_source import (
        MechanismAuthoringSource,
        source_matches_analysis,
    )

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    if not isinstance(mechanism_source, MechanismAuthoringSource):
        raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
    solver_config = solver_config_for_execution_settings(
        execution_settings=execution_settings,
        scientific_settings=scientific_settings,
    )
    normalized_initials: dict[str, float] = {}
    for raw_name, raw_value in initial_values:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("initial value names must be nonblank.")
        if name in normalized_initials:
            raise ValueError(f"initial values contain duplicate name {name!r}.")
        if type(raw_value) is bool or not isinstance(raw_value, Real):
            raise TypeError(f"initial value for {name!r} must be numeric.")
        try:
            normalized_initials[name] = validate_concentration(
                raw_value,
                field=f"initial value for {name!r}",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"initial value for {name!r} must be an admitted concentration."
            ) from exc
    if not normalized_initials:
        raise ValueError("a simulation target must contain initial values.")
    normalized_parameters: dict[str, float] | None = None
    if parameter_values:
        normalized_parameters = {}
        for raw_name, raw_value in parameter_values.items():
            name = str(raw_name or "").strip()
            if not name:
                raise ValueError("parameter value names must be nonblank.")
            if type(raw_value) is bool or not isinstance(raw_value, Real):
                raise TypeError(f"parameter value for {name!r} must be numeric.")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(
                    f"parameter value for {name!r} must be finite."
                )
            normalized_parameters[name] = value
    try:
        t_start = validate_time_seconds(
            execution_settings.t_start,
            field="execution t_start",
        )
        t_end = validate_time_seconds(
            execution_settings.t_end,
            field="execution t_end",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("execution time span must use admitted seconds.") from exc
    if t_end <= t_start:
        raise ValueError("execution t_end must be greater than t_start.")
    try:
        # Realize the generated N grid at the Run/Preview request boundary so
        # binary64-unrepresentable spans fail before worker launch.
        build_time_grid(t_start, t_end, solver_config.grid)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution grid cannot be realized on this time span.") from exc
    if not source_matches_analysis(mechanism_source, prepared_dsl.analysis):
        raise ValueError(
            "prepared DSL source does not match the captured mechanism source."
        )
    if float(prepared_dsl.analysis.temperature_K) != float(
        scientific_settings.temperature
    ):
        raise ValueError(
            "prepared DSL nominal temperature does not match tab science."
        )
    species_names = tuple(str(name) for name in prepared_dsl.analysis.species_names)
    if not species_names:
        raise ValueError("prepared execution requires at least one species.")
    if set(normalized_initials) != set(species_names):
        raise ValueError(
            "direct execution initial values must cover every accepted species exactly once."
        )
    request = SimulationExecutionRequest(
        analysis=prepared_dsl.analysis,
        prepared_dsl=prepared_dsl,
        species_names=species_names,
        y0=tuple(normalized_initials[name] for name in species_names),
        temperature_schedule=prepared_dsl.temperature_schedule_for_execution(),
        parameter_algebra_spec=prepared_dsl.parameter_algebra_spec_for_execution(),
        intervention_execution=prepared_dsl.intervention_execution_contract(),
        t_span=(t_start, t_end),
        solver_config=solver_config,
        wegscheider_admission_report=prepared_dsl.wegscheider_admission_report(),
        parameterized_intervention_schedule=prepared_dsl.parameterized_intervention_schedule_for_binding(),
        requested_parameter_names=(),
        parameter_overrides=normalized_parameters,
        mechanism_source=mechanism_source,
    )
    return bind_simulation_execution_request_interventions(request)


def metadata_view_for_mechanism(
    mechanism: Any,
    *,
    temperature_schedule_override: object = _MISSING,
) -> Any:
    meta_raw = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(meta_raw, Mapping):
        meta_raw = {}
    if temperature_schedule_override is not _MISSING:
        meta_raw = dict(meta_raw)
        meta_raw[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = temperature_schedule_override
    return MechanismMetadataView.from_metadata(meta_raw)


def prepare_preparation_failure(stage: str, message: object) -> SimulationPreparationError:
    return SimulationPreparationError(str(stage or "unknown"), str(message or ""))


def fit_simulation_error_from_preparation_error(
    exc: SimulationPreparationError,
    *,
    failed_params: Optional[Dict[str, float]] = None,
) -> FitSimulationError:
    details = dict(exc.details)
    if exc.parameter_names:
        details.setdefault("value_dependencies", tuple(exc.parameter_names))
        details.setdefault("parameter_names", tuple(exc.parameter_names))
    return FitSimulationError(
        str(exc),
        failed_params=failed_params,
        details=details,
    )


def _coerce_parameter_override_items(
    parameter_overrides: Mapping[str, Any] | None,
    *,
    require_finite: bool = True,
) -> list[tuple[str, float]]:
    if not isinstance(parameter_overrides, Mapping) or not parameter_overrides:
        return []
    items: list[tuple[str, float]] = []
    for raw_name, raw_value in parameter_overrides.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SimulationPreparationError(
                "parameter_overrides",
                f"Parameter override {name!r} must be numeric.",
                parameter_names=(name,),
            ) from exc
        if bool(require_finite) and not np.isfinite(value):
            raise SimulationPreparationError(
                "parameter_overrides",
                f"Parameter override {name!r} must be finite.",
                parameter_names=(name,),
            )
        items.append((name, value))
    return items


def _scalar_parameter_info(mechanism: Any, name: str) -> Mapping[str, Any]:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return {}
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping):
        info = scalar_info.get(str(name))
        if isinstance(info, Mapping):
            return info
    return {}


def scalar_parameter_is_derived(mechanism: Any, name: str) -> bool:
    info = _scalar_parameter_info(mechanism, str(name))
    return bool(info.get("derived"))


def _parameter_algebra_scalar_authority_names(mechanism: Any, spec: Any) -> tuple[frozenset[str], frozenset[str]]:
    assignments = list(getattr(spec, "param_statements", []) or [])
    if not assignments:
        return frozenset(), frozenset()
    from kindred.core.simulator.parameter_algebra import read_mechanism_parameter_values
    from kindred.core.simulator.parameter_algebra_eval import build_parameter_algebra_evaluation_model

    namespace = getattr(spec, "mechanism_namespace", None)
    namespace_info = getattr(namespace, "info_by_name", {}) if namespace is not None else {}
    base_values = read_mechanism_parameter_values(
        mechanism,
        names=set(getattr(spec, "mechanism_param_names", set()) or set()),
    )
    evaluation_model = build_parameter_algebra_evaluation_model(
        spec,
        assignments=assignments,
        base_values=base_values,
    )
    source_owned: set[str] = set()
    derived_scalar: set[str] = set()
    for assignment in assignments:
        name = str(getattr(assignment, "name", "") or "")
        if not name or name in namespace_info:
            continue
        dependencies = set(
            evaluation_model.solver_dependencies_by_assignment.get(name, frozenset())
        )
        if dependencies:
            derived_scalar.add(name)
        else:
            source_owned.add(name)
    return frozenset(source_owned), frozenset(derived_scalar)


def _scalar_schedule_resolution_values_from_mechanism(mechanism: Any) -> Dict[str, float]:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return {}
    values: Dict[str, float] = {}
    scalar_params = meta.get("scalar_params")
    if isinstance(scalar_params, Mapping):
        for raw_name, raw_value in scalar_params.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value):
                values[str(raw_name)] = float(value)
    scalar_bindings = meta.get("scalar_param_bindings")
    if isinstance(scalar_bindings, Mapping):
        for raw_name, binding in scalar_bindings.items():
            try:
                value = float(binding()) if callable(binding) else float(binding)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value):
                values[str(raw_name)] = float(value)
    return values


def _schedule_resolution_values_for_request(
    unresolved_intervention_schedule: InterventionSchedule | None,
    *,
    raw_values: Mapping[str, float] | None = None,
    mechanism_name_by_raw: Mapping[str, str] | None = None,
    unbound_mechanism_name_by_raw: Mapping[str, str] | None = None,
    schedule_name_by_raw: Mapping[str, str] | None = None,
    mechanism: Any | None = None,
) -> Dict[str, float] | None:
    values: Dict[str, float] = {}
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    if mechanism is not None:
        if referenced:
            from kindred.core.simulator.parameter_algebra import read_mechanism_parameter_values

            values.update(
                {
                    str(name): float(value)
                    for name, value in read_mechanism_parameter_values(
                        mechanism,
                        names={str(name) for name in referenced if str(name)},
                    ).items()
                }
            )
            values.update(_scalar_schedule_resolution_values_from_mechanism(mechanism))
    if raw_values is not None:
        values.update({str(name): float(value) for name, value in raw_values.items()})
        for raw_name, canonical_name in {
            **dict(mechanism_name_by_raw or {}),
            **dict(unbound_mechanism_name_by_raw or {}),
        }.items():
            if raw_name in raw_values:
                values[str(canonical_name)] = float(raw_values[raw_name])
        for raw_name, owner_name in dict(schedule_name_by_raw or {}).items():
            if raw_name not in values and owner_name in values:
                values[str(raw_name)] = float(values[owner_name])
        if referenced:
            return values
    return values if values else None


def _scalar_parameter_override_known(mechanism: Any, name: str) -> bool:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return False
    scalar_bindings = meta.get("scalar_param_bindings")
    if isinstance(scalar_bindings, dict) and name in scalar_bindings:
        if scalar_parameter_is_derived(mechanism, name):
            return False
        setter = getattr(scalar_bindings.get(name), "set", None)
        if callable(setter):
            return True
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping) and name in scalar_info:
        return not scalar_parameter_is_derived(mechanism, name)
    scalar_params = meta.get("scalar_params")
    if isinstance(scalar_params, dict) and name in scalar_params:
        if scalar_parameter_is_derived(mechanism, name):
            return False
        return True
    return False


def _prepared_parameter_override_can_apply(mechanism: Any, name: str) -> bool:
    from kindred.core.simulator.step_indexing import lookup_step_param_target

    target_name = _canonical_step_override_name(mechanism, name)
    target = lookup_step_param_target(mechanism, target_name)
    if target is None:
        return _scalar_parameter_override_known(mechanism, name)
    kind, idx, role, entry = target
    if kind == "equilibrium":
        _equilibrium_override_binding_target(
            mechanism,
            idx=int(idx),
            role=str(role),
            entry=entry,
        )
        return True
    candidate = (
        getattr(mechanism.reactions[int(idx)], "rate", None)
        if kind == "reaction" and role == "k"
        else None
    )
    setter = getattr(candidate, "set", None)
    return callable(setter)


def invalid_request_parameter_identifier_message(mechanism: Any, name: str) -> str | None:
    name_s = str(name or "").strip()
    if not name_s:
        return None
    namespace = build_namespace_from_mechanism(mechanism)
    resolution = namespace.resolve(name_s)
    if resolution.canonical_name is not None:
        return None
    if name_s == "K":
        bare_step_key_suggestions = ("Keq1",)
    else:
        bare_step_key_suggestions = {
            "k": ("k1",),
            "kf": ("kf1",),
            "kr": ("kr1",),
            "keq": ("Keq1",),
        }.get(name_s.lower())
    if bare_step_key_suggestions is not None:
        canonical_by_fold = {
            str(canonical).casefold(): str(canonical)
            for canonical in namespace.canonical_names
        }
        suggestions = tuple(
            canonical
            for suggestion in bare_step_key_suggestions
            for canonical in [canonical_by_fold.get(suggestion.casefold())]
            if canonical is not None
        )
        suggestion_text = ", ".join(suggestions)
        suggestion_clause = (
            f"Use canonical indexed parameter name(s): {suggestion_text}. "
            if suggestion_text
            else "Use an existing canonical indexed mechanism parameter, a schedule parameter, or a longer ordinary name. "
        )
        return (
            f"{name_s!r} is not a valid mechanism parameter identifier. "
            f"{suggestion_clause}"
            "Bare step-local DSL keys are not runtime or fitting parameter names."
        )
    invalid_message = namespace.invalid_protected_indexed_identifier_message(name_s)
    if invalid_message is None:
        return None
    return (
        f"{invalid_message} "
        "Protected indexed parameter classes cannot be used as ordinary runtime parameter names."
    )


def _apply_parameter_overrides_to_prepared_mechanism(
    mechanism: Any,
    *,
    parameter_algebra_spec: Any,
    raw_values: Mapping[str, float],
    mechanism_parameter_name_by_raw: Mapping[str, str],
    unbound_mechanism_parameter_name_by_raw: Mapping[str, str],
    unbound_mechanism_parameter_names: Iterable[str],
    scalar_parameter_names: Iterable[str],
    unknown_parameter_names: Iterable[str],
    invalid_parameter_identifier_messages: Mapping[str, str],
) -> None:
    """Apply slider-style values to one isolated execution mechanism."""
    raise_unowned_request_parameter_values(
        raw_values=raw_values,
        mechanism_name_by_raw=mechanism_parameter_name_by_raw,
        unbound_mechanism_name_by_raw=unbound_mechanism_parameter_name_by_raw,
        unbound_mechanism_names=unbound_mechanism_parameter_names,
        unknown_names=unknown_parameter_names,
        invalid_messages=invalid_parameter_identifier_messages,
    )
    if not raw_values:
        return

    mechanism_binding_values = {
        str(canonical_name): float(raw_values[raw_name])
        for raw_name, canonical_name in mechanism_parameter_name_by_raw.items()
        if raw_name in raw_values
    }
    mechanism_binding_values.update(
        {
            str(name): float(raw_values[name])
            for name in scalar_parameter_names
            if name in raw_values
        }
    )
    mechanism_override_items = sorted(
        (str(name), float(value))
        for name, value in mechanism_binding_values.items()
    )
    if not mechanism_override_items:
        return

    from kindred.core.simulator.step_indexing import lookup_step_param_target

    override_applied = False
    for name, value in mechanism_override_items:
        target_name = _canonical_step_override_name(mechanism, name)
        target = lookup_step_param_target(mechanism, target_name)
        if target is None:
            override_applied = (
                _apply_scalar_parameter_override_to_prepared_mechanism(
                    mechanism,
                    name,
                    float(value),
                )
                or override_applied
            )
            continue
        kind, idx, role, entry = target
        target_role = str(role)
        target_value = float(value)
        if kind == "equilibrium":
            target_role, _ = _equilibrium_override_binding_target(
                mechanism,
                idx=int(idx),
                role=str(role),
                entry=entry,
            )
        allow_zero_forward_off = False
        if kind == "equilibrium" and target_role == "kf":
            from kindred.core.equilibrium_rate_authority import (
                EquilibriumRateAuthorityKind,
                normalize_existing_equilibrium_rate_authority,
            )

            allow_zero_forward_off = (
                normalize_existing_equilibrium_rate_authority(
                    mechanism.equilibria[int(idx)]
                ).kind
                == EquilibriumRateAuthorityKind.KEQ
            )
        if (
            (kind == "reaction" and str(role) == "k" and target_value < 0.0)
            or
            (kind == "equilibrium" and target_role in {"kr", "Keq"})
            or (kind == "equilibrium" and target_role == "kf" and not allow_zero_forward_off)
        ) and target_value <= 0.0:
            raise SimulationPreparationError(
                "parameter_overrides",
                f"mechanism parameter {target_name!r} must be positive and finite.",
                parameter_names=(name,),
            )
        candidate = None
        if kind == "reaction" and role == "k":
            candidate = getattr(
                mechanism.reactions[int(idx)],
                "rate",
                None,
            )
        elif kind == "equilibrium" and target_role in {"kf", "kr"}:
            candidate = getattr(
                mechanism.equilibria[int(idx)],
                str(target_role),
                None,
            )
        elif kind == "equilibrium" and target_role == "Keq":
            meta = (
                getattr(
                    mechanism.equilibria[int(idx)],
                    "metadata",
                    {},
                )
                or {}
            )
            if isinstance(meta, Mapping):
                candidate = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)

        setter = getattr(candidate, "set", None)
        if callable(setter):
            setter(float(target_value))
            override_applied = True
    if override_applied:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )

        apply_parameter_algebra_spec_to_mechanism(
            parameter_algebra_spec,
            mechanism=mechanism,
            require_mutable=True,
        )
    return None


def _canonical_step_override_name(mechanism: Any, name: str) -> str:
    name_s = str(name or "").strip()
    if not name_s:
        return ""
    resolution = build_namespace_from_mechanism(mechanism).resolve(name_s)
    if resolution.canonical_name:
        return str(resolution.canonical_name)
    return name_s


def _equilibrium_step_parameter_name(entry: Mapping[str, Any], role: str) -> str:
    try:
        step_index = int(entry.get("step_index"))
    except (TypeError, ValueError) as exc:
        raise SimulationPreparationError(
            "parameter_overrides",
            f"Equilibrium parameter {role!r} is missing a valid canonical step index.",
        ) from exc
    prefix = "Keq" if str(role) == "Keq" else str(role)
    return f"{prefix}{step_index}"


def _equilibrium_override_binding_target(
    mechanism: Any,
    *,
    idx: int,
    role: str,
    entry: Mapping[str, Any],
) -> tuple[str, str]:
    from kindred.core.equilibrium_rate_authority import normalize_existing_equilibrium_rate_authority

    eq = mechanism.equilibria[int(idx)]
    authority = normalize_existing_equilibrium_rate_authority(eq)
    if not authority.authored_role_is_editable(str(role)):
        name = _equilibrium_step_parameter_name(entry, str(role))
        raise SimulationPreparationError(
            "parameter_overrides",
            f"equilibrium parameter {name} is a derived readout and cannot be used as write authority.",
            parameter_names=(name,),
        )
    return str(role), _equilibrium_step_parameter_name(entry, str(role))


def _apply_scalar_parameter_override_to_prepared_mechanism(
    mechanism: Any,
    name: str,
    value: float,
) -> bool:
    from kindred.core.rate_binding import RateBinding

    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return False
    if scalar_parameter_is_derived(mechanism, name):
        raise SimulationPreparationError(
            "parameter_overrides",
            f"derived scalar parameter {name!r} cannot be used as direct runtime override authority.",
            parameter_names=(name,),
        )
    scalar_known = False
    scalar_bindings = meta.get("scalar_param_bindings")
    if not isinstance(scalar_bindings, dict):
        scalar_bindings = {}
        meta["scalar_param_bindings"] = scalar_bindings
    if name in scalar_bindings:
        scalar_known = True
    scalar_params = meta.get("scalar_params")
    if not isinstance(scalar_params, dict):
        scalar_params = {}
        meta["scalar_params"] = scalar_params
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping) and name in scalar_info:
        scalar_known = True
    if name in scalar_params:
        scalar_known = True
    if not scalar_known:
        return False
    binding = scalar_bindings.get(name)
    setter = getattr(binding, "set", None)
    if callable(setter):
        setter(float(value))
    else:
        scalar_bindings[str(name)] = RateBinding(name=str(name), value=float(value))
    scalar_params[str(name)] = float(value)
    return True


def _require_wegscheider_admission_for_execution(
    *,
    admission_report: WegscheiderCyclicityReport | None,
    wegscheider_cyclicity_enabled: bool,
) -> None:
    if not wegscheider_cyclicity_enabled:
        return
    from kindred.core.simulator.wegscheider_symbolic import (
        UnresolvedWegscheiderCyclicityError,
        require_wegscheider_admission_satisfied,
    )

    if admission_report is None:
        raise SimulationPreparationError(
            "wegscheider_admission",
            "Wegscheider-enabled execution requires source admission captured before Algebra.",
        )
    try:
        require_wegscheider_admission_satisfied(admission_report)
    except UnresolvedWegscheiderCyclicityError as exc:
        raise SimulationPreparationError(
            "wegscheider_admission",
            str(exc),
        ) from exc


def _validate_bound_wegscheider_for_execution(
    *,
    mechanism: Any,
    admission_report: WegscheiderCyclicityReport | None,
    wegscheider_cyclicity_enabled: bool,
) -> WegscheiderExecutableSatisfactionReport | None:
    if not wegscheider_cyclicity_enabled:
        return None
    from kindred.core.simulator.wegscheider_symbolic import (
        WegscheiderExecutableSatisfactionError,
        validate_bound_wegscheider_satisfaction,
    )

    if admission_report is None:
        raise SimulationPreparationError(
            "wegscheider_admission",
            "Wegscheider-enabled execution requires source admission captured before mutation.",
        )
    try:
        return validate_bound_wegscheider_satisfaction(
            mechanism,
            admission_report,
        )
    except WegscheiderExecutableSatisfactionError as exc:
        failed_cycles = tuple(
            {
                **cycle.to_payload(),
                "parameter_names": tuple(cycle.parameter_names),
                "authority_parameter_names": tuple(
                    cycle.authority_parameter_names
                ),
            }
            for cycle in exc.cycles
        )
        parameter_names = tuple(
            sorted(
                {
                    name
                    for cycle in exc.cycles
                    for name in (
                        *cycle.parameter_names,
                        *cycle.authority_parameter_names,
                    )
                    if str(name).strip()
                }
            )
        )
        raise SimulationPreparationError(
            "wegscheider_executable",
            str(exc),
            parameter_names=parameter_names,
            details={
                "failed_cycles": failed_cycles,
                "parameter_names": parameter_names,
                "value_dependencies": parameter_names,
                "wegscheider_report": {
                    "cycles": tuple(
                        {
                            "cycle_id": cycle.cycle_id,
                            "step_indices": cycle.step_indices,
                            "equilibrium_indices": cycle.equilibrium_indices,
                            "coefficients": cycle.coefficients,
                            "selected_step_stoichiometric_vectors": (
                                cycle.selected_step_stoichiometric_vectors
                            ),
                        }
                        for cycle in exc.report.admission_report.cycles
                    )
                },
            },
        ) from exc


def _apply_execution_parameter_algebra(
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
    require_mutable: bool,
) -> None:
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
    )
    apply_parameter_algebra_spec_to_mechanism(
        parameter_algebra_spec,
        mechanism=mechanism,
        require_mutable=bool(require_mutable),
    )



def prepare_simulation_worker_run(
    *,
    execution_request: SimulationExecutionRequest,
    progress_callback: Optional[Callable[[float, float, float], None]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
) -> PreparedSimulationRun:
    """
    Prepare an execution-ready SimulationRequest for the GUI SimulationWorker.

    This function is the single owner for:
    - Temperature schedule coercion (DSL schedule takes precedence)
    - Parameter algebra application
    - Metadata flags (e.g. Wegscheider cyclicity enablement)
    - Generated symbolic Jacobian construction when supported by the solver path
    """

    if not isinstance(execution_request, SimulationExecutionRequest):
        raise SimulationPreparationError(
            "execution_request",
            "Worker preparation requires a direct SimulationExecutionRequest.",
        )
    _raise_if_cancelled(cancellation_control)
    t_span = (
        float(execution_request.t_span[0]),
        float(execution_request.t_span[1]),
    )
    prepared_solver_config = execution_request.solver_config
    try:
        # Validate the target grid before worker-owned preparation mutates its
        # one execution shell.
        build_time_grid(t_span[0], t_span[1], prepared_solver_config.grid)
    except (TypeError, ValueError) as exc:
        raise SimulationPreparationError(
            "time_grid",
            "execution grid cannot be realized on this time span.",
        ) from exc
    topology = execution_request.prepared_dsl.compiled_ode_topology()
    symbolic_structure = execution_request.prepared_dsl.symbolic_jacobian_structure()
    _raise_if_cancelled(cancellation_control)

    mechanism = execution_request.prepared_dsl.mutable_mechanism_shell()
    rhs: Callable[..., np.ndarray] | None = None
    species_names = list(execution_request.species_names)
    y0 = np.asarray(execution_request.y0, dtype=float).reshape(-1).copy()
    temperature_schedule_override = execution_request.temperature_schedule
    require_mutable = True

    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = (
            prepared_solver_config.wegscheider_cyclicity_enabled
        )

    _require_wegscheider_admission_for_execution(
        admission_report=execution_request.wegscheider_admission_report,
        wegscheider_cyclicity_enabled=(
            prepared_solver_config.wegscheider_cyclicity_enabled
        ),
    )
    _apply_execution_parameter_algebra(
        mechanism=mechanism,
        parameter_algebra_spec=execution_request.parameter_algebra_spec,
        require_mutable=bool(require_mutable),
    )
    _raise_if_cancelled(cancellation_control)

    unresolved_intervention_schedule = (
        _fresh_preparation_unresolved_intervention_schedule_for_request(
            execution_request
        )
    )
    raw_values: dict[str, float] = {}
    mechanism_names: set[str] = set()
    unbound_mechanism_names: set[str] = set()
    schedule_names: set[str] = set()
    scalar_names: set[str] = set()
    unknown_names: set[str] = set()
    mechanism_name_by_raw: dict[str, str] = {}
    unbound_mechanism_name_by_raw: dict[str, str] = {}
    schedule_name_by_raw: dict[str, str] = {}
    invalid_messages: dict[str, str] = {}
    if (
        bool(execution_request.parameter_overrides)
        or bool(
            unresolved_intervention_schedule is not None
            and unresolved_intervention_schedule.is_parameterized
        )
    ):
        try:
            _bind_request_parameter_overrides(
                mechanism=mechanism,
                parameter_algebra_spec=execution_request.parameter_algebra_spec,
                parameter_overrides=execution_request.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
            )
            _classify_request_parameter_names(
                mechanism=mechanism,
                parameter_overrides=execution_request.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
                raw_values=raw_values,
                mechanism_names=mechanism_names,
                unbound_mechanism_names=unbound_mechanism_names,
                schedule_names=schedule_names,
                scalar_names=scalar_names,
                unknown_names=unknown_names,
                mechanism_name_by_raw=mechanism_name_by_raw,
                unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
                schedule_name_by_raw=schedule_name_by_raw,
                invalid_messages=invalid_messages,
            )
        except SimulationPreparationError:
            raise
    _raise_if_cancelled(cancellation_control)

    wegscheider_executable_report = _validate_bound_wegscheider_for_execution(
        mechanism=mechanism,
        admission_report=execution_request.wegscheider_admission_report,
        wegscheider_cyclicity_enabled=(
            prepared_solver_config.wegscheider_cyclicity_enabled
        ),
    )
    _raise_if_cancelled(cancellation_control)

    intervention_execution = execution_request.intervention_execution
    if unresolved_intervention_schedule is not None:
        if _schedule_parameters_deferred_to_prepared_request(
            execution_request,
            unresolved_intervention_schedule,
        ):
            unresolved_intervention_schedule.validate_species(species_names)
        else:
            try:
                dependency_metadata = _expanded_intervention_schedule_dependency_metadata(
                    unresolved_intervention_schedule,
                    mechanism=mechanism,
                    parameter_algebra_spec=execution_request.parameter_algebra_spec,
                )
                resolved_schedule = _resolve_intervention_schedule_for_request(
                    unresolved_intervention_schedule,
                    parameter_values=_schedule_resolution_values_for_request(
                        unresolved_intervention_schedule,
                        raw_values=raw_values,
                        mechanism_name_by_raw=mechanism_name_by_raw,
                        unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
                        schedule_name_by_raw=schedule_name_by_raw,
                        mechanism=mechanism,
                    ),
                    dependency_metadata=dependency_metadata,
                )
                if resolved_schedule is not None:
                    resolved_schedule.validate_species(species_names)
                intervention_execution = compile_intervention_schedule(
                    resolved_schedule,
                    cancellation_check=(
                        cancellation_control.cancel_requested
                        if cancellation_control is not None
                        else None
                    ),
                    dependency_metadata=dependency_metadata,
                )
                unresolved_intervention_schedule = None
            except (InterventionScheduleError, ValueError) as exc:
                structured_details = dict(getattr(exc, "details", {}) or {})
                structured_dependencies = tuple(
                    getattr(exc, "value_dependencies", ())
                    or structured_details.get("value_dependencies", ())
                )
                if structured_dependencies:
                    structured_details["value_dependencies"] = structured_dependencies
                raise SimulationPreparationError(
                    "intervention_schedule",
                    str(exc),
                    parameter_names=(
                        structured_dependencies
                    ),
                    details=structured_details,
                ) from exc

    if rhs is None:
        try:
            _coefficient_evaluator, rhs = topology.bind_target(mechanism)
        except Exception as exc:
            raise SimulationPreparationError("ode_build", str(exc)) from exc
    _raise_if_cancelled(cancellation_control)

    try:
        temperature_schedule, runtime_symbolic_jacobian, preparation_warnings = build_prepared_run_context(
            mechanism=mechanism,
            temperature_schedule=temperature_schedule_override,
            metadata_view=metadata_view_for_mechanism,
            symbolic_structure=symbolic_structure,
            coefficient_evaluator=_coefficient_evaluator,
        )
    except Exception as exc:
        raise SimulationPreparationError("temperature_schedule", str(exc)) from exc
    _raise_if_cancelled(cancellation_control)

    try:
        initials_for_algebra = {sp: float(y0[idx]) for idx, sp in enumerate(species_names)}
    except Exception as exc:
        raise SimulationPreparationError("initials_for_algebra", str(exc)) from exc

    if not prepared_solver_config.use_analytic_jacobian:
        runtime_symbolic_jacobian = SymbolicJacobianExecution.disabled(
            code="policy-disabled",
            reason="Symbolic Jacobian runtime use disabled by analytic Jacobian policy.",
            jac_sparsity=runtime_symbolic_jacobian.jac_sparsity,
        )
    request: SimulationRequest | SimulationExecutionRequest
    if intervention_execution is None:
        request = execution_request
    else:
        request = SimulationRequest(
            rhs=rhs,
            t_span=(float(t_span[0]), float(t_span[1])),
            y0=np.asarray(y0, dtype=float).reshape(-1),
            intervention_execution=intervention_execution,
            symbolic_jacobian=runtime_symbolic_jacobian,
            solver=prepared_solver_config.solver,
            rtol=prepared_solver_config.rtol,
            atol=prepared_solver_config.atol,
            grid=prepared_solver_config.grid,
            temperature_schedule=temperature_schedule,
            species_names=tuple(species_names),
            progress_callback=progress_callback,
            cancellation_control=cancellation_control,
        )

    return PreparedSimulationRun(
        mechanism=mechanism,
        rhs=rhs,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        species_names=list(species_names),
        temperature_schedule=temperature_schedule,
        parameter_algebra_spec=execution_request.parameter_algebra_spec,
        intervention_execution=intervention_execution,
        symbolic_jacobian=runtime_symbolic_jacobian,
        initials_for_algebra=initials_for_algebra,
        warnings=list(preparation_warnings),
        solver_config=prepared_solver_config,
        request=request,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        cancellation_control=cancellation_control,
        wegscheider_admission_report=execution_request.wegscheider_admission_report,
        wegscheider_executable_report=wegscheider_executable_report,
    )


def _bind_parameters_to_mechanism(mech: Any, names: List[str]) -> None:
    from kindred.core.kinetics import (
        ConstantRateModel,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
    )
    from kindred.core.simulator.step_indexing import lookup_step_param_target
    from kindred.core.validation import try_parse_callable_finite_float

    bindings: Dict[str, RateBinding] = {}
    mechanism_metadata = MechanismMetadataView.from_metadata(
        getattr(mech, "metadata", {}) or {}
    )

    def _forward_binding(
        *,
        name: str,
        model: Any,
    ) -> RateBinding:
        if isinstance(model, ConstantRateModel):
            return RateBinding(name=name, value=float(model.k))
        return ForwardRateBinding(
            name=name,
            model=model,
            reference_temperature_K=float(
                mechanism_metadata.temperature_K
            ),
        )

    for raw_name in names:
        name = str(raw_name or "").strip()
        target = lookup_step_param_target(mech, name)
        if target is None:
            continue
        kind, idx, role, entry = target
        target_name = name
        target_role = str(role)
        if kind == "equilibrium":
            target_role, target_name = _equilibrium_override_binding_target(
                mech,
                idx=int(idx),
                role=str(role),
                entry=entry,
            )

        binding = bindings.get(target_name)
        if binding is None:
            if kind == "reaction" and role == "k":
                rxn = mech.reactions[idx]
                current = getattr(rxn, "rate", None)
                model = forward_rate_model_for_reaction(
                    rxn,
                    default_standard_conc_M=float(
                        mechanism_metadata.standard_conc_M
                    ),
                )
                if isinstance(current, ForwardRateBinding) or (
                    isinstance(model, ConstantRateModel)
                    and isinstance(current, RateBinding)
                ):
                    binding = current
                else:
                    binding = _forward_binding(
                        name=target_name,
                        model=model,
                    )
                    if isinstance(current, RateBinding) and isinstance(
                        binding, ForwardRateBinding
                    ):
                        binding.set(float(current()))
            elif kind == "equilibrium" and target_role in {"kf", "kr"}:
                eq = mech.equilibria[idx]
                current = getattr(eq, target_role, None)
                if target_role == "kr" and isinstance(current, RateBinding):
                    binding = current
                elif target_role == "kf":
                    model = forward_rate_model_for_equilibrium(
                        eq,
                        temperature_K=float(
                            mechanism_metadata.temperature_K
                        ),
                        default_standard_conc_M=float(
                            mechanism_metadata.standard_conc_M
                        ),
                    )
                    if isinstance(current, ForwardRateBinding) or (
                        isinstance(model, ConstantRateModel)
                        and isinstance(current, RateBinding)
                    ):
                        binding = current
                    else:
                        binding = _forward_binding(
                            name=target_name,
                            model=model,
                        )
                        if isinstance(current, RateBinding) and isinstance(
                            binding, ForwardRateBinding
                        ):
                            binding.set(float(current()))
                else:
                    parsed, ok = try_parse_callable_finite_float(current)
                    if not ok or float(parsed) <= 0.0:
                        raise SimulationPreparationError(
                            "parameter_binding",
                            f"Cannot bind {target_name!r}: equilibrium reverse rate must be positive and finite.",
                        )
                    binding = RateBinding(
                        name=target_name,
                        value=float(parsed),
                    )
            elif kind == "equilibrium" and target_role == "Keq":
                eq = mech.equilibria[idx]
                meta = getattr(eq, "metadata", {}) or {}
                current = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
                if isinstance(current, RateBinding):
                    binding = current
                else:
                    parsed, ok = try_parse_callable_finite_float(current)
                    if not ok:
                        parsed, ok = try_parse_callable_finite_float(
                            getattr(eq, "Keq", None)
                        )
                    if not ok:
                        raise SimulationPreparationError(
                            "parameter_binding",
                            (
                                f"Cannot bind {target_name!r}: equilibrium "
                                "parameter has no explicit equilibrium-constant "
                                "source token."
                            ),
                        )
                    if float(parsed) <= 0.0:
                        raise SimulationPreparationError(
                            "parameter_binding",
                            f"Cannot bind {target_name!r}: equilibrium constant must be positive and finite.",
                        )
                    binding = RateBinding(
                        name=target_name,
                        value=float(parsed),
                    )
            else:
                raise SimulationPreparationError(
                    "parameter_binding",
                    f"Cannot bind unsupported mechanism coordinate {target_name!r}.",
                )
            bindings[target_name] = binding

        if kind == "reaction" and role == "k":
            rxn = mech.reactions[idx]
            mech.reactions[idx] = replace(rxn, rate=binding)
        elif kind == "equilibrium" and target_role in {"kf", "kr"}:
            eq = mech.equilibria[idx]
            mech.equilibria[idx] = replace(eq, **{target_role: binding})
        elif kind == "equilibrium" and target_role == "Keq":
            eq = mech.equilibria[idx]
            meta = dict(getattr(eq, "metadata", {}) or {})
            meta[EquilibriumMetadataKeys.KEQ_INPUT] = binding
            mech.equilibria[idx] = replace(eq, metadata=meta)


def _bind_request_parameter_overrides(
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
    parameter_overrides: Mapping[str, float] | None,
    unresolved_intervention_schedule: Any,
) -> None:
    """Bind one request through the fixed preparation-owned parameter path."""

    raw_values: dict[str, float] = {}
    mechanism_names: set[str] = set()
    unbound_mechanism_names: set[str] = set()
    schedule_names: set[str] = set()
    scalar_names: set[str] = set()
    unknown_names: set[str] = set()
    mechanism_name_by_raw: dict[str, str] = {}
    unbound_mechanism_name_by_raw: dict[str, str] = {}
    schedule_name_by_raw: dict[str, str] = {}
    invalid_messages: dict[str, str] = {}
    _classify_request_parameter_names(
        mechanism=mechanism,
        parameter_overrides=parameter_overrides,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        raw_values=raw_values,
        mechanism_names=mechanism_names,
        unbound_mechanism_names=unbound_mechanism_names,
        schedule_names=schedule_names,
        scalar_names=scalar_names,
        unknown_names=unknown_names,
        mechanism_name_by_raw=mechanism_name_by_raw,
        unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
        schedule_name_by_raw=schedule_name_by_raw,
        invalid_messages=invalid_messages,
    )
    mechanism_override_names = sorted(
        set(mechanism_names) | set(unbound_mechanism_names)
    )
    internal_names = _internal_parameter_algebra_binding_names(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
    )
    _bind_parameters_to_mechanism(
        mechanism,
        sorted(set(mechanism_override_names) | set(internal_names)),
    )
    raw_values.clear()
    mechanism_names.clear()
    unbound_mechanism_names.clear()
    schedule_names.clear()
    scalar_names.clear()
    unknown_names.clear()
    mechanism_name_by_raw.clear()
    unbound_mechanism_name_by_raw.clear()
    schedule_name_by_raw.clear()
    invalid_messages.clear()
    _classify_request_parameter_names(
        mechanism=mechanism,
        parameter_overrides=parameter_overrides,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        raw_values=raw_values,
        mechanism_names=mechanism_names,
        unbound_mechanism_names=unbound_mechanism_names,
        schedule_names=schedule_names,
        scalar_names=scalar_names,
        unknown_names=unknown_names,
        mechanism_name_by_raw=mechanism_name_by_raw,
        unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
        schedule_name_by_raw=schedule_name_by_raw,
        invalid_messages=invalid_messages,
    )
    _apply_parameter_overrides_to_prepared_mechanism(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
        raw_values=raw_values,
        mechanism_parameter_name_by_raw=mechanism_name_by_raw,
        unbound_mechanism_parameter_name_by_raw=unbound_mechanism_name_by_raw,
        unbound_mechanism_parameter_names=unbound_mechanism_names,
        scalar_parameter_names=scalar_names,
        unknown_parameter_names=unknown_names,
        invalid_parameter_identifier_messages=invalid_messages,
    )


def _internal_parameter_algebra_binding_names(
    mechanism: Any,
    *,
    parameter_algebra_spec: Any,
) -> set[str]:
    spec = parameter_algebra_spec
    if spec is None:
        raise SimulationPreparationError(
            "parameter_algebra",
            "Parameter algebra binding requires its retained typed spec.",
        )
    mechanism_namespace = (
        spec.mechanism_namespace
        if spec is not None and getattr(spec, "mechanism_namespace", None) is not None
        else build_namespace_from_mechanism(mechanism)
    )
    namespace_info = mechanism_namespace.info_by_name
    constrained = {
        str(stmt.name)
        for stmt in ((getattr(spec, "param_statements", None) or ()) if spec is not None else ())
        if str(stmt.name) in namespace_info
    }
    constrained_keq_targets = {
        name for name in constrained if namespace_info[name].role == "Keq"
    }
    return constrained - constrained_keq_targets


def reject_requested_algebra_owned_mechanism_parameters_for_fitting(
    mechanism: Any,
    parameter_algebra_spec: Any,
    requested_names: Iterable[str] = (),
) -> None:
    requested = {
        _canonical_step_override_name(mechanism, str(name))
        for name in (requested_names or ())
        if str(name or "").strip()
    }
    namespace_info = build_namespace_from_mechanism(mechanism).info_by_name
    algebra_owned = sorted(
        str(name)
        for name in (
            requested
            & _internal_parameter_algebra_binding_names(
                mechanism,
                parameter_algebra_spec=parameter_algebra_spec,
            )
        )
        if str(name) in namespace_info and namespace_info[str(name)].role in {"k", "kf", "kr"}
    )
    if not algebra_owned:
        return
    raise SimulationPreparationError(
        "parameter_algebra",
        (
            "Requested algebra-owned mechanism parameter(s) cannot be fitted directly "
            "because parameter algebra overwrites them: "
            + ", ".join(algebra_owned)
        ),
    )
