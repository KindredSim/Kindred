"""
Simulation preparation utilities used by fitting and prepared/bound execution paths.

The parser-owned ``DslAnalysis`` and ``PreparedDslRequestSet`` enter here as
admitted inputs; this module coordinates typed parameter binding, solver/Jacobian
preparation, and RHS compilation for the execution paths that consume them.
"""

from __future__ import annotations

import copy
import logging
import math
from numbers import Real
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.exceptions import FitSimulationError, SimulationCancelled
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.temperature import (
    TemperatureScheduleProtocol,
    coerce_temperature_schedule,
)
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    InterventionScheduleError,
    coerce_intervention_schedule,
    intervention_schedule_parameter_names,
)
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
    compile_intervention_schedule,
)
from kindred.core.runtime_defaults import (
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.solver_types import (
    SimulationRequest,
    SimulationCancellationControl,
)
from kindred.core.symbolic.identity import normalize_symbolic_identity_mapping
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
)
from kindred.core.simulator.dsl_parse import (
    DslAnalysis,
    PreparedDslRequestSet,
)
from kindred.core.simulator.executable_identity import (
    BaseExecutableScienceIdentity,
    TargetExecutableScienceIdentity,
    compose_target_executable_identity,
)
from kindred.core.simulation_preparation_stages import (
    PreparedSolverConfig as _PreparedSolverConfig,
    build_prepared_run_context,
    build_solver_config,
)

logger = logging.getLogger(__name__)


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
    "BoundMechanism",
    "SimulationExecutionSnapshot",
    "SimulationExecutionRequest",
    "PreparedSimulationMetadata",
    "PreparedSimulationRun",
    "SimulationParameterValuePartition",
    "SimulationPreparationError",
    "build_simulation_request_from_prepared_run",
    "metadata_view_for_mechanism",
    "prepare_bound_mechanism",
    "capture_simulation_execution_snapshot",
    "compose_simulation_target_identity",
    "bind_simulation_execution_request_interventions",
    "build_simulation_execution_request",
    "solver_config_for_execution_settings",
    "prepare_simulation_worker_run",
    "partition_simulation_parameter_values",
    "resolve_prepared_run_intervention_execution",
    "canonicalize_request_parameter_names",
    "fit_simulation_error_from_preparation_error",
    "prepare_preparation_failure",
    "raise_unowned_request_parameter_values",
    "reject_requested_algebra_owned_mechanism_parameters_for_fitting",
    "scalar_parameter_is_derived",
]


@dataclass(frozen=True)
class SimulationParameterValuePartition:
    """Typed ownership partition for request-time parameter override names."""

    raw_values: Mapping[str, float]
    mechanism_parameter_names: frozenset[str] = frozenset()
    unbound_mechanism_parameter_names: frozenset[str] = frozenset()
    schedule_parameter_names: frozenset[str] = frozenset()
    scalar_parameter_names: frozenset[str] = frozenset()
    unknown_parameter_names: frozenset[str] = frozenset()
    mechanism_parameter_name_by_raw: Mapping[str, str] = field(default_factory=dict)
    unbound_mechanism_parameter_name_by_raw: Mapping[str, str] = field(default_factory=dict)
    schedule_parameter_name_by_raw: Mapping[str, str] = field(default_factory=dict)
    invalid_parameter_identifier_messages: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "raw_values",
            "mechanism_parameter_name_by_raw",
            "unbound_mechanism_parameter_name_by_raw",
            "schedule_parameter_name_by_raw",
            "invalid_parameter_identifier_messages",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping.")
            object.__setattr__(self, name, _freeze_prepared_value(value))

    @property
    def schedule_only_parameter_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in self.schedule_parameter_names
            if self.schedule_parameter_name_by_raw.get(name, name) not in self.mechanism_parameter_names
            and self.schedule_parameter_name_by_raw.get(name, name) not in self.unbound_mechanism_parameter_names
            and name not in self.scalar_parameter_names
        )

    @property
    def bindable_mechanism_parameter_names(self) -> frozenset[str]:
        return frozenset(set(self.mechanism_parameter_names) | set(self.unbound_mechanism_parameter_names))

    @property
    def mechanism_binding_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for raw_name, canonical_name in self.mechanism_parameter_name_by_raw.items():
            if raw_name in self.raw_values:
                out[str(canonical_name)] = float(self.raw_values[raw_name])
        for name in self.scalar_parameter_names:
            if name in self.raw_values:
                out[str(name)] = float(self.raw_values[name])
        return out

    @property
    def schedule_resolution_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {str(name): float(value) for name, value in self.raw_values.items()}
        for raw_name, canonical_name in {
            **self.mechanism_parameter_name_by_raw,
            **self.unbound_mechanism_parameter_name_by_raw,
        }.items():
            if raw_name in self.raw_values:
                out[str(canonical_name)] = float(self.raw_values[raw_name])
        for schedule_raw_name, owner_name in self.schedule_parameter_name_by_raw.items():
            if schedule_raw_name in out:
                continue
            if owner_name in out:
                out[str(schedule_raw_name)] = float(out[owner_name])
        return out


def _parameter_override_target_kind(mechanism: Any, name: str) -> str | None:
    from kindred.core.simulator.step_indexing import lookup_step_param_target

    target_name = _canonical_step_override_name(mechanism, name)
    if lookup_step_param_target(mechanism, target_name) is not None:
        return "mechanism" if _prepared_parameter_override_can_apply(mechanism, name) else "unbound_mechanism"
    if _scalar_parameter_override_known(mechanism, name):
        return "scalar"
    return None


def partition_simulation_parameter_values(
    *,
    mechanism: Any,
    parameter_overrides: Mapping[str, Any] | None,
    unresolved_intervention_schedule: InterventionSchedule | None,
    requested_parameter_names: Iterable[str] | None = None,
    scalar_parameter_names: Iterable[str] | None = None,
    validate_values: bool = True,
) -> SimulationParameterValuePartition:
    raw_values = {
        str(name): float(value)
        for name, value in _coerce_parameter_override_items(
            parameter_overrides,
            require_finite=bool(validate_values),
        )
    }
    classification_names = list(raw_values)
    for requested_name in requested_parameter_names or ():
        name = str(requested_name or "").strip()
        if name and name not in classification_names:
            classification_names.append(name)
    schedule_names = frozenset(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    for schedule_name in schedule_names:
        name = str(schedule_name or "").strip()
        if name and name not in classification_names:
            classification_names.append(name)
    if not raw_values and not classification_names:
        return SimulationParameterValuePartition(raw_values={})
    declared_scalar_names = {
        str(name or "").strip()
        for name in scalar_parameter_names or ()
        if str(name or "").strip()
    }
    mechanism_namespace = build_namespace_from_mechanism(mechanism)
    mechanism_names: set[str] = set()
    unbound_mechanism_names: set[str] = set()
    scalar_names: set[str] = set()
    unknown_names: set[str] = set()
    mechanism_name_by_raw: dict[str, str] = {}
    unbound_mechanism_name_by_raw: dict[str, str] = {}
    schedule_name_by_raw: dict[str, str] = {}
    invalid_messages: dict[str, str] = {}

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

    return SimulationParameterValuePartition(
        raw_values=raw_values,
        mechanism_parameter_names=frozenset(mechanism_names),
        unbound_mechanism_parameter_names=frozenset(unbound_mechanism_names),
        schedule_parameter_names=schedule_names,
        scalar_parameter_names=frozenset(scalar_names),
        unknown_parameter_names=frozenset(unknown_names),
        mechanism_parameter_name_by_raw=mechanism_name_by_raw,
        unbound_mechanism_parameter_name_by_raw=unbound_mechanism_name_by_raw,
        schedule_parameter_name_by_raw=schedule_name_by_raw,
        invalid_parameter_identifier_messages=invalid_messages,
    )


def raise_unowned_request_parameter_values(
    parameter_partition: SimulationParameterValuePartition,
    *,
    allow_unbound_mechanism_parameters: bool = False,
) -> None:
    if parameter_partition.invalid_parameter_identifier_messages:
        first_name = sorted(parameter_partition.invalid_parameter_identifier_messages)[0]
        raise ValueError(parameter_partition.invalid_parameter_identifier_messages[first_name])
    owner_to_raw: dict[str, str] = {}
    for raw_name, owner_name in sorted(
        {
            **parameter_partition.mechanism_parameter_name_by_raw,
            **parameter_partition.unbound_mechanism_parameter_name_by_raw,
        }.items()
    ):
        if raw_name not in parameter_partition.raw_values:
            continue
        owner = str(owner_name)
        previous = owner_to_raw.get(owner)
        if previous is not None and previous != raw_name:
            raise ValueError(
                f"Request parameter names {previous!r} and {raw_name!r} both resolve to {owner!r}."
            )
        owner_to_raw[owner] = str(raw_name)
    unknown_names = sorted(str(name) for name in parameter_partition.unknown_parameter_names)
    if unknown_names:
        raise ValueError(
            "Unknown request parameter(s): "
            + ", ".join(repr(name) for name in unknown_names)
            + ". Request parameters must belong to the mechanism, schedule, scalar algebra, or initial-condition namespace."
        )
    if parameter_partition.unbound_mechanism_parameter_names and not allow_unbound_mechanism_parameters:
        first_name = sorted(parameter_partition.unbound_mechanism_parameter_names)[0]
        raw_names = sorted(
            raw_name
            for raw_name, canonical_name in parameter_partition.unbound_mechanism_parameter_name_by_raw.items()
            if canonical_name == first_name
        )
        raw_clause = f" from request name {raw_names[0]!r}" if raw_names else ""
        raise ValueError(
            f"Request parameter{raw_clause} resolves to mechanism parameter {first_name!r}, "
            "but no prepared execution binding can consume it."
        )


def canonicalize_request_parameter_names(
    parameter_partition: SimulationParameterValuePartition,
    requested_parameter_names: Iterable[str] | None,
) -> list[str]:
    canonical_by_raw = {
        **parameter_partition.mechanism_parameter_name_by_raw,
        **parameter_partition.unbound_mechanism_parameter_name_by_raw,
    }
    out: list[str] = []
    owner_to_raw: dict[str, str] = {}
    for raw_name in requested_parameter_names or ():
        raw = str(raw_name or "").strip()
        if not raw:
            continue
        owner = str(canonical_by_raw.get(raw, raw))
        previous = owner_to_raw.get(owner)
        if previous is not None and previous != raw:
            raise ValueError(
                f"Duplicate request parameter names {previous!r} and {raw!r} resolve to {owner!r}."
            )
        owner_to_raw[owner] = raw
        out.append(owner)
    return out


@dataclass
class BoundMechanism:
    """Prepared Fit mechanism and its retained typed execution authorities."""

    mechanism: Any
    species_names: List[str]
    y0: np.ndarray
    dsl_analysis: DslAnalysis
    parameter_algebra_spec: Any
    base_executable_identity: BaseExecutableScienceIdentity
    intervention_execution: InterventionExecutionContract | None = None
    unresolved_intervention_schedule: InterventionSchedule | None = None


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
        candidate_domain_eligible: bool = False,
    ) -> None:
        super().__init__(str(message))
        self.stage = str(stage or "unknown")
        self.parameter_names = frozenset(
            str(name).strip()
            for name in parameter_names
            if str(name).strip()
        )
        self.candidate_domain_eligible = bool(candidate_domain_eligible)


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
    symbolic_wegscheider_identity: Mapping[str, Any] | None
    initials_for_algebra: Optional[Mapping[str, float]]
    warnings: tuple[str, ...]
    solver_config: _PreparedSolverConfig
    request: SimulationRequest | SimulationExecutionRequest
    unresolved_intervention_schedule: InterventionSchedule | None = None
    cancellation_control: SimulationCancellationControl | None = None

    def __post_init__(self) -> None:
        y0 = np.asarray(self.y0, dtype=float).reshape(-1).copy()
        y0.setflags(write=False)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "species_names", tuple(self.species_names))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
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
        if self.initials_for_algebra is not None:
            if not isinstance(self.initials_for_algebra, Mapping):
                raise TypeError("initials_for_algebra must be a mapping or None.")
            object.__setattr__(
                self,
                "initials_for_algebra",
                _freeze_prepared_value(self.initials_for_algebra),
            )
        if self.symbolic_wegscheider_identity is not None:
            if not isinstance(self.symbolic_wegscheider_identity, Mapping):
                raise TypeError(
                    "symbolic_wegscheider_identity must be a mapping or None."
                )
            object.__setattr__(
                self,
                "symbolic_wegscheider_identity",
                _freeze_prepared_value(self.symbolic_wegscheider_identity),
            )


def resolve_prepared_run_intervention_execution(
    prepared_run: PreparedSimulationRun,
    parameter_partition: SimulationParameterValuePartition,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> InterventionExecutionContract:
    """Return the one resolved Run-owned contract consumed by Fit."""

    schedule = prepared_run.unresolved_intervention_schedule
    if schedule is None or not schedule.is_parameterized:
        return prepared_run.intervention_execution
    try:
        resolved_schedule = _resolve_intervention_schedule_for_request(
            schedule,
            parameter_values=_schedule_resolution_values_for_request(
                schedule,
                parameter_partition,
                mechanism=prepared_run.mechanism,
            ),
        )
        return compile_intervention_schedule(
            resolved_schedule,
            cancellation_check=cancellation_check,
        )
    except InterventionScheduleError as exc:
        raise SimulationPreparationError(
            "intervention_schedule",
            str(exc),
            parameter_names=parameter_partition.schedule_parameter_names,
            candidate_domain_eligible=True,
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
        symbolic_wegscheider_identity=request.symbolic_wegscheider_identity,
        cancellation_control=cancellation_control,
    )
    _raise_if_cancelled(cancellation_control)
    return result


@dataclass(frozen=True)
class PreparedSimulationMetadata:
    """Typed metadata attached to prepared simulation closures for fitting workflows."""

    version: int
    mechanism_text_sha256: str
    mechanism_text_len: int
    param_names: tuple[str, ...]
    t_end: float
    num_points: int
    temperature_K: float
    solver_config: _PreparedSolverConfig
    initial_prefix: str
    intervention_schedule_declarative_fingerprint: str = ""
    intervention_schedule_executable_fingerprint: str = ""
    symbolic_jacobian_identity: Optional[Mapping[str, Any]] = None
    symbolic_jacobian_status: Optional[Mapping[str, Any]] = None
    symbolic_wegscheider_identity: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver_config, _PreparedSolverConfig):
            raise TypeError(
                "prepared simulation metadata requires PreparedSolverConfig."
            )
        object.__setattr__(
            self,
            "param_names",
            tuple(str(name) for name in self.param_names if str(name).strip()),
        )
        jacobian_identity = normalize_symbolic_identity_mapping(
            self.symbolic_jacobian_identity,
            label="symbolic Jacobian identity",
        )
        object.__setattr__(
            self,
            "symbolic_jacobian_identity",
            None
            if jacobian_identity is None
            else _freeze_prepared_value(jacobian_identity),
        )
        if self.symbolic_jacobian_status is not None:
            if not isinstance(self.symbolic_jacobian_status, Mapping):
                raise TypeError("symbolic_jacobian_status must be a mapping or None.")
            object.__setattr__(
                self,
                "symbolic_jacobian_status",
                _freeze_prepared_value(self.symbolic_jacobian_status),
            )
        wegscheider_identity = normalize_symbolic_identity_mapping(
            self.symbolic_wegscheider_identity,
            label="symbolic Wegscheider identity",
        )
        object.__setattr__(
            self,
            "symbolic_wegscheider_identity",
            None
            if wegscheider_identity is None
            else _freeze_prepared_value(wegscheider_identity),
        )

    def to_serializable_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": int(self.version),
            "mechanism_text_sha256": str(self.mechanism_text_sha256),
            "mechanism_text_len": int(self.mechanism_text_len),
            "param_names": [str(name) for name in self.param_names if str(name).strip()],
            "t_end": float(self.t_end),
            "num_points": int(self.num_points),
            "temperature_K": float(self.temperature_K),
            "solver_requested": self.solver_config.solver_input,
            "solver_normalized": self.solver_config.solver,
            "rtol": self.solver_config.rtol,
            "atol": self.solver_config.atol,
            "use_analytic_jacobian": (
                self.solver_config.use_analytic_jacobian
            ),
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: (
                self.solver_config.wegscheider_cyclicity_enabled
            ),
            "initial_prefix": str(self.initial_prefix),
            "intervention_schedule_declarative_fingerprint": str(
                self.intervention_schedule_declarative_fingerprint or ""
            ),
            "intervention_schedule_executable_fingerprint": str(
                self.intervention_schedule_executable_fingerprint or ""
            ),
        }
        if self.symbolic_jacobian_identity:
            payload["symbolic_jacobian_identity"] = _mutable_prepared_value(
                self.symbolic_jacobian_identity
            )
        if self.symbolic_jacobian_status:
            payload["symbolic_jacobian_status"] = _mutable_prepared_value(
                self.symbolic_jacobian_status
            )
        if self.symbolic_wegscheider_identity:
            payload["symbolic_wegscheider_identity"] = _mutable_prepared_value(
                self.symbolic_wegscheider_identity
            )
        return payload

def compose_simulation_target_identity(
    base_identity: BaseExecutableScienceIdentity,
    intervention_execution: InterventionExecutionContract,
) -> TargetExecutableScienceIdentity:
    """Compose identity at the direct target-binding authority."""

    return compose_target_executable_identity(
        base_identity,
        intervention_execution,
    )


@dataclass(frozen=True, slots=True)
class SimulationExecutionSnapshot:
    """One isolated, typed scientific state captured for direct execution."""

    analysis: DslAnalysis
    mechanism: Any
    species_names: tuple[str, ...]
    y0: tuple[float, ...]
    temperature_schedule: TemperatureScheduleProtocol | None
    parameter_algebra_spec: Any
    intervention_execution: InterventionExecutionContract | None
    base_executable_identity: BaseExecutableScienceIdentity
    target_executable_identity: TargetExecutableScienceIdentity | None
    parameterized_intervention_schedule: InterventionSchedule | None = None
    requested_parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.analysis) is not DslAnalysis:
            raise TypeError("execution snapshot analysis must be DslAnalysis.")
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
                "execution snapshot species_names must be unique nonblank strings."
            )
        y0 = tuple(float(value) for value in self.y0)
        if len(y0) != len(species_names) or any(
            not np.isfinite(value) or value < 0.0 for value in y0
        ):
            raise ValueError(
                "execution snapshot y0 must contain one finite nonnegative "
                "value per species."
            )
        if self.temperature_schedule is not None:
            coerce_temperature_schedule(self.temperature_schedule)
        if self.parameter_algebra_spec is None:
            raise TypeError(
                "execution snapshot requires its retained parameter algebra spec."
            )
        if not isinstance(
            self.base_executable_identity,
            BaseExecutableScienceIdentity,
        ):
            raise TypeError(
                "execution snapshot requires admitted base executable identity."
            )
        if self.base_executable_identity.analysis_identity != self.analysis.identity:
            raise ValueError(
                "execution snapshot analysis and base executable identities conflict."
            )
        schedule = self.parameterized_intervention_schedule
        contract = self.intervention_execution
        if schedule is not None:
            if type(schedule) is not InterventionSchedule or not schedule.is_parameterized:
                raise TypeError(
                    "execution snapshot declarative intervention state must be "
                    "an exact parameterized schedule."
                )
            if contract is not None:
                raise ValueError(
                    "execution snapshot intervention state cannot be both "
                    "declarative and compiled."
                )
            if self.target_executable_identity is not None:
                raise ValueError(
                    "unbound execution snapshot cannot carry target identity."
                )
        elif not isinstance(contract, InterventionExecutionContract):
            raise TypeError(
                "bound execution snapshots require one compiled intervention "
                "contract."
            )
        else:
            expected_identity = compose_simulation_target_identity(
                self.base_executable_identity,
                contract,
            )
            if self.target_executable_identity != expected_identity:
                raise ValueError(
                    "bound execution snapshot target identity conflicts with "
                    "its admitted base and intervention contract."
                )
        requested_names = tuple(
            str(name).strip() for name in self.requested_parameter_names
        )
        if any(not name for name in requested_names) or len(
            requested_names
        ) != len(set(requested_names)):
            raise ValueError(
                "execution snapshot requested parameter names must be unique "
                "and nonblank."
            )
        object.__setattr__(self, "species_names", species_names)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "requested_parameter_names", requested_names)

    def isolated_copy(self) -> "SimulationExecutionSnapshot":
        return replace(
            self,
            mechanism=_clone_prepared_authority_mechanism(self.mechanism),
            temperature_schedule=copy.deepcopy(self.temperature_schedule),
            parameter_algebra_spec=copy.deepcopy(self.parameter_algebra_spec),
            parameterized_intervention_schedule=copy.deepcopy(
                self.parameterized_intervention_schedule
            ),
        )


@dataclass(frozen=True, slots=True)
class SimulationExecutionRequest:
    """Exact direct request for one isolated simulation target."""

    snapshot: SimulationExecutionSnapshot
    t_span: tuple[float, float]
    solver_config: _PreparedSolverConfig
    parameter_overrides: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SimulationExecutionSnapshot):
            raise TypeError(
                "execution request snapshot must be SimulationExecutionSnapshot."
            )
        if type(self.t_span) is not tuple or len(self.t_span) != 2:
            raise TypeError("execution request t_span must be an exact pair.")
        t_span = (float(self.t_span[0]), float(self.t_span[1]))
        if (
            not np.isfinite(t_span[0])
            or not np.isfinite(t_span[1])
            or t_span[0] < 0.0
            or t_span[1] <= t_span[0]
        ):
            raise ValueError(
                "execution request requires finite 0 <= t_start < t_end."
            )
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
        object.__setattr__(self, "t_span", t_span)
        object.__setattr__(self, "parameter_overrides", overrides)

    def isolated_copy(self) -> "SimulationExecutionRequest":
        return type(self)(
            snapshot=self.snapshot.isolated_copy(),
            t_span=self.t_span,
            solver_config=self.solver_config,
            parameter_overrides=self.parameter_overrides,
        )

_MISSING = object()


def _fresh_preparation_unresolved_intervention_schedule_for_request(
    execution_request: SimulationExecutionRequest,
) -> InterventionSchedule | None:
    return copy.deepcopy(
        execution_request.snapshot.parameterized_intervention_schedule
    )


def _resolve_intervention_schedule_for_request(
    schedule: InterventionSchedule | None,
    *,
    parameter_values: Mapping[str, Any] | None = None,
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
        return schedule.resolve_parameters(values)
    except InterventionScheduleError as exc:
        raise SimulationPreparationError(
            "intervention_schedule",
            str(exc),
            parameter_names=intervention_schedule_parameter_names(schedule),
            candidate_domain_eligible=True,
        ) from exc


def _schedule_parameters_deferred_to_prepared_request(
    execution_request: SimulationExecutionRequest,
    unresolved_intervention_schedule: InterventionSchedule,
) -> frozenset[str]:
    if not unresolved_intervention_schedule.is_parameterized:
        return frozenset()
    requested = execution_request.snapshot.requested_parameter_names
    supplied = set(dict(execution_request.parameter_overrides or {}).keys())
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    return frozenset(
        str(name)
        for name in requested
        if str(name).strip()
        and str(name) in referenced
        and str(name) not in supplied
    )


def bind_simulation_execution_request_interventions(
    request: SimulationExecutionRequest,
) -> SimulationExecutionRequest:
    """Bind and compile one parameterized direct target exactly once."""

    if not isinstance(request, SimulationExecutionRequest):
        raise TypeError(
            "intervention binding requires SimulationExecutionRequest."
        )
    snapshot = request.snapshot
    unresolved = snapshot.parameterized_intervention_schedule
    if unresolved is None:
        return request
    if _schedule_parameters_deferred_to_prepared_request(request, unresolved):
        return request
    mechanism = _clone_prepared_authority_mechanism(snapshot.mechanism)
    prepared_solver_config = request.solver_config
    metadata = getattr(mechanism, "metadata", None)
    if isinstance(metadata, dict):
        metadata[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = (
            prepared_solver_config.wegscheider_cyclicity_enabled
        )
    _apply_execution_parameter_algebra_and_cyclicity(
        mechanism=mechanism,
        parameter_algebra_spec=snapshot.parameter_algebra_spec,
        prepared_solver_config=prepared_solver_config,
        require_mutable=True,
    )
    try:
        parameter_partition = _bind_request_parameter_overrides(
            mechanism=mechanism,
            parameter_algebra_spec=snapshot.parameter_algebra_spec,
            parameter_overrides=request.parameter_overrides,
            unresolved_intervention_schedule=unresolved,
        )
        resolved = _resolve_intervention_schedule_for_request(
            unresolved,
            parameter_values=_schedule_resolution_values_for_request(
                unresolved,
                parameter_partition,
                mechanism=mechanism,
            ),
        )
        if resolved is not None:
            resolved.validate_species(snapshot.species_names)
        contract = compile_intervention_schedule(resolved)
    except SimulationPreparationError:
        raise
    except (InterventionScheduleError, ValueError) as exc:
        raise SimulationPreparationError(
            "intervention_schedule",
            str(exc),
        ) from exc
    schedule_only_names = parameter_partition.schedule_only_parameter_names
    remaining_overrides = {
        str(name): float(value)
        for name, value in dict(request.parameter_overrides or {}).items()
        if str(name) not in schedule_only_names
    }
    return SimulationExecutionRequest(
        snapshot=replace(
            snapshot,
            parameterized_intervention_schedule=None,
            intervention_execution=contract,
            target_executable_identity=compose_simulation_target_identity(
                snapshot.base_executable_identity,
                contract,
            ),
        ),
        t_span=request.t_span,
        solver_config=request.solver_config,
        parameter_overrides=remaining_overrides or None,
    )


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
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"initial value for {name!r} must be finite and nonnegative."
            )
        normalized_initials[name] = value
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
    t_start = float(execution_settings.t_start)
    t_end = float(execution_settings.t_end)
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
    return bind_simulation_execution_request_interventions(
        SimulationExecutionRequest(
            snapshot=capture_simulation_execution_snapshot(
                prepared_dsl,
                initial_values=normalized_initials,
            ),
            t_span=(t_start, t_end),
            solver_config=solver_config,
            parameter_overrides=normalized_parameters,
        )
    )


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


def capture_simulation_execution_snapshot(
    prepared_dsl: PreparedDslRequestSet,
    *,
    initial_values: Mapping[str, float] | None = None,
    requested_parameter_names: Iterable[str] = (),
) -> SimulationExecutionSnapshot:
    """Capture one complete direct execution snapshot from prepared authority."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
    mechanism = prepared_dsl.mechanism_for_execution()
    species_names = tuple(str(name) for name in mechanism.species_names())
    if not species_names:
        raise ValueError("prepared execution requires at least one species.")
    if initial_values is None:
        values = {
            name: float(mechanism.species[name].initial_conc)
            for name in species_names
        }
    else:
        values = {
            str(name): float(value)
            for name, value in initial_values.items()
        }
        if set(values) != set(species_names):
            raise ValueError(
                "direct execution initial values must cover every accepted "
                "species exactly once."
            )
    temperature_schedule = metadata_view_for_mechanism(
        mechanism
    ).temperature_schedule
    base_identity = prepared_dsl.base_executable_identity()
    intervention_execution = prepared_dsl.intervention_execution_contract()
    return SimulationExecutionSnapshot(
        analysis=prepared_dsl.analysis,
        mechanism=mechanism,
        species_names=species_names,
        y0=tuple(values[name] for name in species_names),
        temperature_schedule=temperature_schedule,
        parameter_algebra_spec=(
            prepared_dsl.parameter_algebra_spec_for_execution()
        ),
        intervention_execution=intervention_execution,
        base_executable_identity=base_identity,
        target_executable_identity=(
            None
            if intervention_execution is None
            else compose_simulation_target_identity(
                base_identity,
                intervention_execution,
            )
        ),
        parameterized_intervention_schedule=(
            prepared_dsl.parameterized_intervention_schedule_for_binding()
        ),
        requested_parameter_names=tuple(requested_parameter_names),
    )


def prepare_preparation_failure(stage: str, message: object) -> SimulationPreparationError:
    return SimulationPreparationError(str(stage or "unknown"), str(message or ""))


def _clone_prepared_authority_mechanism(mechanism: Any) -> Any:
    clone_method = getattr(mechanism, "clone", None)
    if callable(clone_method):
        return clone_method()
    return copy.deepcopy(mechanism)


def _preparation_failure_payload(exc: SimulationPreparationError) -> Dict[str, Any]:
    from kindred.core.simulation_failure import build_simulation_failure

    return build_simulation_failure(
        "preparation_error",
        str(exc),
        details={
            "stage": str(exc.stage or "unknown"),
            "parameter_names": tuple(sorted(exc.parameter_names)),
        },
        exc_type=exc.__class__.__name__,
    )


def fit_simulation_error_from_preparation_error(
    exc: SimulationPreparationError,
    *,
    failed_params: Optional[Dict[str, float]] = None,
) -> FitSimulationError:
    return FitSimulationError(
        str(exc),
        failed_params=failed_params,
        details={
            "fatal": True,
            "candidate_domain": False,
            "stage": exc.stage,
            "failure": _preparation_failure_payload(exc),
        },
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
    parameter_partition: SimulationParameterValuePartition | None,
    *,
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
    if parameter_partition is not None:
        values.update(parameter_partition.schedule_resolution_values)
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
        suggestions = tuple(
            canonical
            for suggestion in bare_step_key_suggestions
            for canonical in [namespace.canonical_by_lower.get(suggestion.lower())]
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
    parameter_partition: SimulationParameterValuePartition,
) -> None:
    """Apply slider-style values to one isolated execution mechanism."""
    partition = parameter_partition
    raise_unowned_request_parameter_values(partition)
    if not partition.raw_values:
        return

    mechanism_override_items = sorted(
        (str(name), float(value))
        for name, value in partition.mechanism_binding_values.items()
    )
    if not mechanism_override_items:
        return

    from kindred.core.simulator.step_indexing import lookup_step_param_target

    override_values_by_name = {str(name): float(value) for name, value in mechanism_override_items}
    _reject_conflicting_equilibrium_reverse_overrides(mechanism, override_values_by_name)
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
        if (
            (kind == "reaction" and str(role) == "k")
            or (
                kind == "equilibrium"
                and target_role in {"kf", "kr", "Keq"}
            )
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


def _finite_runtime_override_scalar(value: object, *, label: str) -> float:
    from kindred.core.validation import try_parse_callable_finite_float

    parsed, ok = try_parse_callable_finite_float(value)
    if not ok or not math.isfinite(float(parsed)):
        raise SimulationPreparationError("parameter_overrides", f"{label} must be finite.")
    return float(parsed)


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


def _reject_conflicting_equilibrium_reverse_overrides(
    mechanism: Any,
    override_values_by_name: Mapping[str, float],
) -> None:
    if not override_values_by_name:
        return
    from kindred.core.equilibrium_rate_authority import (
        effective_reverse_rate_from_keq,
        normalize_existing_equilibrium_rate_authority,
    )
    from kindred.core.simulator.step_indexing import get_step_index_map

    requested = {str(name): float(value) for name, value in override_values_by_name.items()}
    for entry in get_step_index_map(mechanism):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            idx = int(entry.get("equilibrium_index"))
        except (TypeError, ValueError):
            continue
        kf_name = _equilibrium_step_parameter_name(entry, "kf")
        kr_name = _equilibrium_step_parameter_name(entry, "kr")
        keq_name = _equilibrium_step_parameter_name(entry, "Keq")
        if kr_name not in requested or keq_name not in requested:
            continue
        eq = mechanism.equilibria[int(idx)]
        authority = normalize_existing_equilibrium_rate_authority(eq)
        kf_value = (
            float(requested[kf_name])
            if kf_name in requested
            else _finite_runtime_override_scalar(getattr(eq, "kf", None), label=f"equilibrium parameter {kf_name}")
        )
        reverse_std_ratio = _finite_runtime_override_scalar(
            authority.reverse_std_ratio,
            label=f"equilibrium {keq_name} reverse standard ratio",
        )
        if reverse_std_ratio <= 0.0:
            raise SimulationPreparationError(
                "parameter_overrides",
                f"equilibrium {keq_name} reverse standard ratio must be positive.",
                parameter_names=(keq_name,),
            )
        keq_value = float(requested[keq_name])
        if keq_value <= 0.0:
            raise SimulationPreparationError(
                "parameter_overrides",
                f"equilibrium parameter {keq_name} must be positive.",
                parameter_names=(keq_name,),
            )
        expected_kr = float(effective_reverse_rate_from_keq(kf_value, keq_value, reverse_std_ratio))
        actual_kr = float(requested[kr_name])
        if not math.isclose(actual_kr, expected_kr, rel_tol=1e-9, abs_tol=1e-12):
            raise SimulationPreparationError(
                "parameter_overrides",
                (
                    f"Conflicting equilibrium parameter overrides for {kr_name} and {keq_name}: "
                    f"{kr_name}={actual_kr:.17g} is inconsistent with "
                    f"{keq_name}={keq_value:.17g} and {kf_name}={float(kf_value):.17g}."
                ),
                parameter_names=(kf_name, kr_name, keq_name),
            )


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


def _apply_execution_parameter_algebra_and_cyclicity(
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
    prepared_solver_config: _PreparedSolverConfig,
    require_mutable: bool,
) -> None:
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
    )
    from kindred.core.simulator.wegscheider_symbolic import (
        UnresolvedWegscheiderCyclicityError,
    )

    try:
        apply_parameter_algebra_spec_to_mechanism(
            parameter_algebra_spec,
            mechanism=mechanism,
            require_mutable=bool(require_mutable),
        )
    except UnresolvedWegscheiderCyclicityError as exc:
        raise SimulationPreparationError("wegscheider_cyclicity", str(exc)) from exc



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
    snapshot = execution_request.snapshot.isolated_copy()
    _raise_if_cancelled(cancellation_control)
    prepared_solver_config = execution_request.solver_config

    mechanism = snapshot.mechanism
    rhs: Callable[..., np.ndarray] | None = None
    species_names = list(snapshot.species_names)
    y0 = np.asarray(snapshot.y0, dtype=float).reshape(-1).copy()
    temperature_schedule_override = snapshot.temperature_schedule
    require_mutable = True

    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = (
            prepared_solver_config.wegscheider_cyclicity_enabled
        )

    _apply_execution_parameter_algebra_and_cyclicity(
        mechanism=mechanism,
        parameter_algebra_spec=snapshot.parameter_algebra_spec,
        prepared_solver_config=prepared_solver_config,
        require_mutable=bool(require_mutable),
    )
    _raise_if_cancelled(cancellation_control)

    unresolved_intervention_schedule = (
        _fresh_preparation_unresolved_intervention_schedule_for_request(
            execution_request
        )
    )
    parameter_partition: SimulationParameterValuePartition | None = None
    if (
        bool(execution_request.parameter_overrides)
        or bool(
            unresolved_intervention_schedule is not None
            and unresolved_intervention_schedule.is_parameterized
        )
    ):
        try:
            parameter_partition = _bind_request_parameter_overrides(
                mechanism=mechanism,
                parameter_algebra_spec=snapshot.parameter_algebra_spec,
                parameter_overrides=execution_request.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
            )
        except SimulationPreparationError:
            raise
    _raise_if_cancelled(cancellation_control)

    intervention_execution = snapshot.intervention_execution
    if unresolved_intervention_schedule is not None:
        deferred_schedule_parameters = (
            _schedule_parameters_deferred_to_prepared_request(
                execution_request,
                unresolved_intervention_schedule,
            )
        )
        if deferred_schedule_parameters:
            unresolved_intervention_schedule.validate_species(species_names)
        else:
            try:
                resolved_schedule = _resolve_intervention_schedule_for_request(
                    unresolved_intervention_schedule,
                    parameter_values=_schedule_resolution_values_for_request(
                        unresolved_intervention_schedule,
                        parameter_partition,
                        mechanism=mechanism,
                    ),
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
                )
                unresolved_intervention_schedule = None
            except (InterventionScheduleError, ValueError) as exc:
                raise SimulationPreparationError(
                    "intervention_schedule",
                    str(exc),
                    parameter_names=(
                        ()
                        if parameter_partition is None
                        else parameter_partition.schedule_parameter_names
                    ),
                    candidate_domain_eligible=True,
                ) from exc

    if rhs is None:
        try:
            rhs = build_ode_rhs_from_mechanism(mechanism)
        except Exception as exc:
            raise SimulationPreparationError("ode_build", str(exc)) from exc
    _raise_if_cancelled(cancellation_control)

    try:
        prepared_context = build_prepared_run_context(
            mechanism=mechanism,
            solver_config=prepared_solver_config,
            temperature_schedule=temperature_schedule_override,
            metadata_view=metadata_view_for_mechanism,
        )
    except Exception as exc:
        raise SimulationPreparationError("temperature_schedule", str(exc)) from exc
    _raise_if_cancelled(cancellation_control)

    temperature_schedule = prepared_context.temperature_schedule
    symbolic_wegscheider_identity = None
    meta = getattr(mechanism, "metadata", {}) or {}
    if isinstance(meta, Mapping) and isinstance(meta.get("symbolic_wegscheider_identity"), Mapping):
        symbolic_wegscheider_identity = dict(meta.get("symbolic_wegscheider_identity") or {})

    try:
        initials_for_algebra = {sp: float(y0[idx]) for idx, sp in enumerate(species_names)}
    except Exception as exc:
        raise SimulationPreparationError("initials_for_algebra", str(exc)) from exc

    runtime_symbolic_jacobian = prepared_context.symbolic_jacobian
    if (
        prepared_solver_config.solver.upper() in {"BDF", "RADAU"}
        and not prepared_solver_config.use_analytic_jacobian
    ):
        runtime_symbolic_jacobian = SymbolicJacobianExecution.disabled(
            code="policy-disabled",
            reason="Symbolic Jacobian runtime use disabled by analytic Jacobian policy.",
            jac_sparsity=prepared_context.symbolic_jacobian.jac_sparsity,
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
            symbolic_wegscheider_identity=symbolic_wegscheider_identity,
            cancellation_control=cancellation_control,
        )

    return PreparedSimulationRun(
        mechanism=mechanism,
        rhs=rhs,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        species_names=list(species_names),
        temperature_schedule=temperature_schedule,
        parameter_algebra_spec=snapshot.parameter_algebra_spec,
        intervention_execution=intervention_execution,
        symbolic_jacobian=runtime_symbolic_jacobian,
        symbolic_wegscheider_identity=symbolic_wegscheider_identity,
        initials_for_algebra=initials_for_algebra,
        warnings=list(prepared_context.warnings),
        solver_config=prepared_solver_config,
        request=request,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        cancellation_control=cancellation_control,
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
) -> SimulationParameterValuePartition:
    """Bind one request through the fixed preparation-owned parameter path."""

    parameter_partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides=parameter_overrides,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )
    mechanism_override_names = sorted(
        parameter_partition.bindable_mechanism_parameter_names
    )
    internal_names = _internal_parameter_algebra_binding_names(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
    )
    _bind_parameters_to_mechanism(
        mechanism,
        sorted(set(mechanism_override_names) | set(internal_names)),
    )
    parameter_partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides=parameter_overrides,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )
    _apply_parameter_overrides_to_prepared_mechanism(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
        parameter_partition=parameter_partition,
    )
    return parameter_partition


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


def prepare_bound_mechanism(
    prepared_dsl: PreparedDslRequestSet,
    param_names: List[str],
    *,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
) -> BoundMechanism:
    """
    Bind selected parameters on one retained typed DSL authority.

    Returns a BoundMechanism holding mutable RateBinding objects so callers can
    update parameter values without parsing or rebuilding the ODE system.
    """
    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    if type(wegscheider_cyclicity_enabled) is not bool:
        raise TypeError("wegscheider_cyclicity_enabled must be a bool.")
    mechanism = prepared_dsl.mechanism_for_execution()
    unresolved_intervention_schedule = coerce_intervention_schedule(
        prepared_dsl.parameterized_intervention_schedule_for_binding()
    )
    spec = prepared_dsl.parameter_algebra_spec_for_execution()
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED
        ] = wegscheider_cyclicity_enabled
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

    try:
        mechanism_namespace = build_namespace_from_mechanism(mechanism)
        namespace_info = mechanism_namespace.info_by_name
        declared_scalar_names, derived_scalar_names = _parameter_algebra_scalar_authority_names(mechanism, spec)
        requested_derived_scalar_names = sorted(
            name for name in (param_names or []) if str(name) in derived_scalar_names
        )
        if requested_derived_scalar_names:
            raise SimulationPreparationError(
                "parameter_binding",
                "derived scalar parameter(s) cannot be used as fitting write authority: "
                + ", ".join(repr(name) for name in requested_derived_scalar_names),
            )
        requested_parameter_partition = partition_simulation_parameter_values(
            mechanism=mechanism,
            parameter_overrides=None,
            unresolved_intervention_schedule=unresolved_intervention_schedule,
            requested_parameter_names=param_names or [],
            scalar_parameter_names=declared_scalar_names,
        )
        raise_unowned_request_parameter_values(
            requested_parameter_partition,
            allow_unbound_mechanism_parameters=True,
        )
        constrained = {
            stmt.name
            for stmt in (spec.param_statements or [])
            if str(stmt.name) in namespace_info
        }
        constrained_keq_targets = {
            str(name) for name in constrained if namespace_info[str(name)].role == "Keq"
        }
        constrained_mutable_targets = set(constrained) - constrained_keq_targets
        requested = {
            _canonical_step_override_name(mechanism, str(name))
            for name in requested_parameter_partition.bindable_mechanism_parameter_names
            if str(name or "").strip()
        }
        requested_dependent_keq = sorted(str(name) for name in (requested & constrained_keq_targets))
        if requested_dependent_keq:
            raise SimulationPreparationError(
                "parameter_algebra",
                (
                    "Dependent equilibrium parameter(s) cannot be fitted directly "
                    "because Wegscheider/algebra constraints overwrite them: "
                    + ", ".join(requested_dependent_keq)
                ),
            )
        mech_bind_names = sorted(
            {
                str(n)
                for n in (
                    (requested - constrained_keq_targets)
                    | constrained_mutable_targets
                )
                if str(n) in namespace_info
            }
        )
    except Exception as exc:
        raise fit_simulation_error_from_preparation_error(
            prepare_preparation_failure("parameter_algebra", exc)
        ) from exc

    _bind_parameters_to_mechanism(mechanism, mech_bind_names)
    try:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )
        from kindred.core.simulator.wegscheider_symbolic import UnresolvedWegscheiderCyclicityError

        apply_parameter_algebra_spec_to_mechanism(
            spec,
            mechanism=mechanism,
            require_mutable=True,
        )
    except UnresolvedWegscheiderCyclicityError as exc:
        raise fit_simulation_error_from_preparation_error(
            prepare_preparation_failure("wegscheider_cyclicity", exc)
        ) from exc
    except Exception as exc:
        raise fit_simulation_error_from_preparation_error(
            prepare_preparation_failure("parameter_algebra", exc)
        ) from exc

    bound = BoundMechanism(
        mechanism=mechanism,
        species_names=species_names,
        y0=y0,
        dsl_analysis=prepared_dsl.analysis,
        parameter_algebra_spec=spec,
        base_executable_identity=prepared_dsl.base_executable_identity(),
        intervention_execution=(
            prepared_dsl.intervention_execution_contract()
        ),
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )

    return bound
