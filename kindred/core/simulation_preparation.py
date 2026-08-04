"""
Simulation preparation utilities used by fitting and prepared/bound execution paths.

This module intentionally owns the "parse DSL → bind parameters → compile RHS" pipeline
so that optimization code and analysis code can depend on a narrower surface area.
"""

from __future__ import annotations

import copy
import logging
import math
import pickle
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.exceptions import FitSimulationError
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.temperature import (
    TemperatureScheduleDictProtocol,
    TemperatureScheduleProtocol,
    coerce_temperature_schedule,
    temperature_schedule_execution_identity,
)
from kindred.core.time_grid import normalize_time_grid_policy
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    InterventionScheduleError,
    coerce_intervention_schedule,
    intervention_schedule_identity_fingerprints,
    intervention_schedule_parameter_names,
    normalized_intervention_schedule_payload,
)
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
    compile_intervention_schedule,
)
from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.simulator.solvers import (
    DEFAULT_SOLVER_NAME,
    SimulationRequest,
    normalize_solver_name,
)
from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.identity import normalize_symbolic_identity_mapping
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.symbolic.namespaces import symbolic_status_payload
from kindred.core.symbolic.structure_cache import (
    get_or_build_symbolic_jacobian_structure,
    symbolic_jacobian_structure_cache_key,
)
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
)
from kindred.core.simulator.dsl_parse import (
    DslAnalysis,
    PreparedDslRequestSet,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoundMechanism",
    "SimulationExecutionRequest",
    "PreparedSimulationMetadata",
    "PreparedSimulationRun",
    "SimulationParameterValuePartition",
    "SimulationPreparationError",
    "assert_simulation_execution_request_schedule_identity",
    "build_simulation_request_from_prepared_run",
    "coerce_prepared_simulation_metadata",
    "metadata_view_for_mechanism",
    "prepare_bound_mechanism",
    "prepare_simulation_worker_run",
    "materialize_request_intervention_schedule_for_parameter_values",
    "intervention_schedule_identity_for_execution_request",
    "materialized_intervention_schedule_for_execution_request",
    "partition_simulation_parameter_values",
    "resolve_prepared_run_intervention_execution",
    "canonicalize_request_parameter_names",
]


@dataclass(frozen=True)
class _ParameterOverrideApplication:
    rebuild_rhs: bool


@dataclass(frozen=True)
class SimulationParameterValuePartition:
    """Typed ownership partition for request-time parameter override names."""

    raw_values: Dict[str, float]
    mechanism_parameter_names: frozenset[str] = frozenset()
    unbound_mechanism_parameter_names: frozenset[str] = frozenset()
    schedule_parameter_names: frozenset[str] = frozenset()
    scalar_parameter_names: frozenset[str] = frozenset()
    runtime_parameter_names: frozenset[str] = frozenset()
    unknown_parameter_names: frozenset[str] = frozenset()
    mechanism_parameter_name_by_raw: Dict[str, str] = field(default_factory=dict)
    unbound_mechanism_parameter_name_by_raw: Dict[str, str] = field(default_factory=dict)
    schedule_parameter_name_by_raw: Dict[str, str] = field(default_factory=dict)
    runtime_parameter_name_by_raw: Dict[str, str] = field(default_factory=dict)
    invalid_parameter_identifier_messages: Dict[str, str] = field(default_factory=dict)

    @property
    def schedule_only_parameter_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in self.schedule_parameter_names
            if self.schedule_parameter_name_by_raw.get(name, name) not in self.mechanism_parameter_names
            and self.schedule_parameter_name_by_raw.get(name, name) not in self.unbound_mechanism_parameter_names
            and name not in self.scalar_parameter_names
            and name not in self.runtime_parameter_names
        )

    @property
    def mechanism_binding_names(self) -> frozenset[str]:
        return frozenset(
            set(self.mechanism_parameter_names)
            | set(self.scalar_parameter_names)
            | set(self.runtime_parameter_names)
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
        for raw_name, runtime_name in self.runtime_parameter_name_by_raw.items():
            if raw_name in self.raw_values:
                out[str(runtime_name)] = float(self.raw_values[raw_name])
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
    runtime_parameter_names: Iterable[str] | None = None,
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
    declared_runtime_names = {
        str(name or "").strip()
        for name in runtime_parameter_names or ()
        if str(name or "").strip()
    }
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    mechanism_namespace = build_namespace_from_mechanism(mechanism)
    mechanism_names: set[str] = set()
    unbound_mechanism_names: set[str] = set()
    scalar_names: set[str] = set()
    runtime_names: set[str] = set()
    unknown_names: set[str] = set()
    mechanism_name_by_raw: dict[str, str] = {}
    unbound_mechanism_name_by_raw: dict[str, str] = {}
    schedule_name_by_raw: dict[str, str] = {}
    runtime_name_by_raw: dict[str, str] = {}
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
        if name in raw_values and _scalar_parameter_is_derived(mechanism, name):
            invalid_messages[name] = (
                f"derived scalar parameter {name!r} cannot be used as direct runtime "
                "or schedule override authority."
            )
            continue
        if name in schedule_names:
            schedule_name_by_raw[name] = name
            claimed = True
        if name in declared_runtime_names:
            runtime_names.add(name)
            runtime_name_by_raw[name] = name
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
        runtime_parameter_names=frozenset(runtime_names),
        unknown_parameter_names=frozenset(unknown_names),
        mechanism_parameter_name_by_raw=mechanism_name_by_raw,
        unbound_mechanism_parameter_name_by_raw=unbound_mechanism_name_by_raw,
        schedule_parameter_name_by_raw=schedule_name_by_raw,
        runtime_parameter_name_by_raw=runtime_name_by_raw,
        invalid_parameter_identifier_messages=invalid_messages,
    )


def _raise_unowned_request_parameter_values(
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
    """Precompiled mechanism with mutable rate bindings."""

    mechanism: Any
    rhs: Callable[..., np.ndarray]
    bindings: Dict[str, Any]
    species_names: List[str]
    y0: np.ndarray
    param_names: List[str]
    dsl_analysis: DslAnalysis
    unresolved_intervention_schedule: InterventionSchedule | None = None

    def as_serializable_execution_payload(self) -> Mapping[str, Any]:
        """Return the sole process-safe typed execution payload."""
        mechanism_snapshot = _clone_prepared_authority_mechanism(self.mechanism)
        metadata = metadata_view_for_mechanism(mechanism_snapshot)
        intervention_schedule = getattr(metadata, "intervention_schedule", None)
        payload: dict[str, Any] = {
            "version": 4,
            "dsl_analysis": self.dsl_analysis,
            "mechanism_blob": pickle.dumps(
                mechanism_snapshot,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
            "y0": np.array(self.y0, copy=True),
            "species_names": list(self.species_names),
            "temperature_schedule": metadata.temperature_schedule,
            "intervention_schedule": (
                intervention_schedule.to_payload()
                if intervention_schedule is not None
                else None
            ),
            "unresolved_intervention_schedule": (
                self.unresolved_intervention_schedule.to_payload()
                if self.unresolved_intervention_schedule is not None
                else None
            ),
            "jacobian_func": None,
        }
        return payload


class SimulationPreparationError(RuntimeError):
    """
    Error raised when preparing a SimulationWorker run payload/request fails.

    The GUI worker maps `stage` values to user-facing messages while keeping the
    preparation logic in core.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(str(message))
        self.stage = str(stage or "unknown")


def _cancellation_check_from_events(
    events: Iterable[Callable[..., object]] | None,
) -> Callable[[], bool] | None:
    callbacks = tuple(
        callback
        for event in tuple(events or ())
        if bool(getattr(event, "_kindred_cancel_event", False))
        for callback in (getattr(event, "_kindred_cancelled", None),)
        if callable(callback)
    )
    if not callbacks:
        return None

    def cancellation_requested() -> bool:
        return any(bool(callback()) for callback in callbacks)

    return cancellation_requested


@dataclass(frozen=True)
class PreparedSimulationRun:
    """Execution-ready simulation payload produced by core preparation utilities."""

    mechanism: Any
    rhs: Callable[..., np.ndarray]
    y0: np.ndarray
    species_names: List[str]
    solver_input: str
    temperature_schedule: TemperatureScheduleProtocol | None
    intervention_execution: InterventionExecutionContract
    jacobian_func: Any
    jac_sparsity: Any
    initials_for_algebra: Optional[Dict[str, float]]
    warnings: List[str]
    request: SimulationRequest | SimulationExecutionRequest
    unresolved_intervention_schedule: InterventionSchedule | None = None


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
            None,
            parameter_values=_schedule_resolution_values_for_request(
                None,
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
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc


def build_simulation_request_from_prepared_run(
    prepared_run: PreparedSimulationRun,
    *,
    y0: np.ndarray,
    intervention_execution: InterventionExecutionContract,
    symbolic_jacobian: SymbolicJacobianExecution,
    events: Iterable[Callable[..., object]] | None = None,
) -> SimulationRequest:
    request = prepared_run.request
    if not isinstance(request, SimulationRequest):
        raise SimulationPreparationError(
            "simulation_request",
            "Prepared run solver request assembly requires a solver SimulationRequest.",
        )
    event_values = tuple(events or ())
    return SimulationRequest(
        rhs=request.rhs,
        t_span=tuple(map(float, request.t_span)),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        intervention_execution=intervention_execution,
        solver=str(request.solver),
        rtol=float(request.rtol),
        atol=float(request.atol),
        grid=dict(request.grid or {}),
        **symbolic_jacobian.to_request_kwargs(),
        temperature_schedule=request.temperature_schedule,
        species_names=tuple(prepared_run.species_names),
        events=event_values if event_values else None,
        symbolic_wegscheider_identity=request.symbolic_wegscheider_identity,
    )


@dataclass(frozen=True)
class PreparedSimulationMetadata:
    """Typed metadata attached to prepared simulation closures for fitting workflows."""

    version: int
    mechanism_text_sha256: str
    mechanism_text_len: int
    param_names: List[str]
    t_end: float
    num_points: int
    temperature_K: float
    solver_requested: str
    solver_normalized: str
    rtol: float
    atol: float
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool
    initial_prefix: str
    intervention_schedule_declarative_fingerprint: str = ""
    intervention_schedule_executable_fingerprint: str = ""
    symbolic_jacobian_identity: Optional[Dict[str, Any]] = None
    symbolic_jacobian_status: Optional[Dict[str, Any]] = None
    symbolic_wegscheider_identity: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbolic_jacobian_identity",
            normalize_symbolic_identity_mapping(
                self.symbolic_jacobian_identity,
                label="symbolic Jacobian identity",
            ),
        )
        object.__setattr__(
            self,
            "symbolic_wegscheider_identity",
            normalize_symbolic_identity_mapping(
                self.symbolic_wegscheider_identity,
                label="symbolic Wegscheider identity",
            ),
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
            "solver_requested": str(self.solver_requested),
            "solver_normalized": str(self.solver_normalized),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "use_sparse_jacobian": bool(self.use_sparse_jacobian),
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: bool(
                self.wegscheider_cyclicity_enabled
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
            payload["symbolic_jacobian_identity"] = dict(self.symbolic_jacobian_identity)
        if self.symbolic_jacobian_status:
            payload["symbolic_jacobian_status"] = dict(self.symbolic_jacobian_status)
        if self.symbolic_wegscheider_identity:
            payload["symbolic_wegscheider_identity"] = dict(self.symbolic_wegscheider_identity)
        return payload

    @classmethod
    def from_mapping(cls, meta: Mapping[str, Any]) -> "PreparedSimulationMetadata":
        if "intervention_schedule_fingerprint" in meta:
            raise KeyError("stale intervention_schedule_fingerprint")
        for key in (
            "intervention_schedule_declarative_fingerprint",
            "intervention_schedule_executable_fingerprint",
        ):
            if key not in meta:
                raise KeyError(key)
        param_names_raw = meta.get("param_names") or []
        if not isinstance(param_names_raw, (list, tuple)):
            param_names_raw = []
        if "solver_requested" not in meta:
            raise KeyError("solver_requested")
        if "solver_normalized" not in meta:
            raise KeyError("solver_normalized")
        solver_requested = str(meta.get("solver_requested") or "").strip()
        solver_normalized = str(meta.get("solver_normalized") or "").strip()
        return cls(
            version=int(meta.get("version") or 1),
            mechanism_text_sha256=str(meta.get("mechanism_text_sha256") or ""),
            mechanism_text_len=int(meta.get("mechanism_text_len") or 0),
            param_names=sorted({str(x) for x in param_names_raw if str(x).strip()}),
            t_end=float(meta.get("t_end") or 0.0),
            num_points=int(meta.get("num_points") or 0),
            temperature_K=float(meta.get("temperature_K") or 0.0),
            solver_requested=solver_requested,
            solver_normalized=solver_normalized,
            rtol=float(meta.get("rtol") or 0.0),
            atol=float(meta.get("atol") or 0.0),
            use_sparse_jacobian=bool(
                meta.get("use_sparse_jacobian", USE_SPARSE_JACOBIAN_DEFAULT)
            ),
            wegscheider_cyclicity_enabled=bool(
                meta.get(
                    MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                )
            ),
            initial_prefix=str(meta.get("initial_prefix") or ""),
            intervention_schedule_declarative_fingerprint=str(
                meta.get("intervention_schedule_declarative_fingerprint") or ""
            ),
            intervention_schedule_executable_fingerprint=str(
                meta.get("intervention_schedule_executable_fingerprint") or ""
            ),
            symbolic_jacobian_identity=meta.get("symbolic_jacobian_identity"),
            symbolic_jacobian_status=(
                dict(meta.get("symbolic_jacobian_status") or {})
                if isinstance(meta.get("symbolic_jacobian_status"), Mapping)
                else None
            ),
            symbolic_wegscheider_identity=meta.get("symbolic_wegscheider_identity"),
        )


_INTERVENTION_SCHEDULE_UNSET = object()


class _ScheduleAuthorityState(Enum):
    ABSENT = "absent"
    EXPLICIT_NONE = "explicit_none"
    EXPLICIT_SCHEDULE = "explicit_schedule"


@dataclass(frozen=True)
class _ScheduleAuthorityDecision:
    state: _ScheduleAuthorityState
    unresolved_schedule: InterventionSchedule | None = None


@dataclass(frozen=True, init=False)
class SimulationExecutionRequest:
    """Structured execution handoff for worker and batch simulation paths."""

    prepared_payload: Optional[Mapping[str, Any]]
    initials: Dict[str, float]
    t_span: Tuple[float, float]
    solver_config: Dict[str, Any]
    simulation_identity: Optional[Dict[str, Any]] = None
    parameter_overrides: Optional[Dict[str, float]] = None
    intervention_schedule: InterventionSchedule | Mapping[str, Any] | None = None
    version: int = 1
    _intervention_schedule_authority: bool = field(default=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        prepared_payload: Optional[Mapping[str, Any]],
        initials: Mapping[str, Any],
        t_span: Tuple[float, float],
        solver_config: Mapping[str, Any],
        simulation_identity: Optional[Mapping[str, Any]] = None,
        parameter_overrides: Optional[Mapping[str, Any]] = None,
        intervention_schedule: InterventionSchedule | Mapping[str, Any] | None | object = _INTERVENTION_SCHEDULE_UNSET,
        version: int = 1,
    ) -> None:
        schedule_authority = intervention_schedule is not _INTERVENTION_SCHEDULE_UNSET
        stored_schedule = None if intervention_schedule is _INTERVENTION_SCHEDULE_UNSET else intervention_schedule
        object.__setattr__(self, "prepared_payload", dict(prepared_payload) if isinstance(prepared_payload, Mapping) else None)
        object.__setattr__(self, "initials", {str(name): float(value) for name, value in dict(initials or {}).items()})
        object.__setattr__(self, "t_span", (float(t_span[0]), float(t_span[1])))
        object.__setattr__(self, "solver_config", dict(solver_config or {}))
        object.__setattr__(self, "simulation_identity", dict(simulation_identity or {}) if simulation_identity else None)
        object.__setattr__(
            self,
            "parameter_overrides",
            {str(name): float(value) for name, value in dict(parameter_overrides or {}).items()}
            if isinstance(parameter_overrides, Mapping)
            else None,
        )
        object.__setattr__(self, "intervention_schedule", stored_schedule)
        object.__setattr__(self, "version", int(version or 1))
        object.__setattr__(self, "_intervention_schedule_authority", bool(schedule_authority))

    @property
    def has_intervention_schedule_authority(self) -> bool:
        return bool(self._intervention_schedule_authority)

    def with_intervention_schedule(
        self,
        intervention_schedule: InterventionSchedule | Mapping[str, Any] | None,
    ) -> "SimulationExecutionRequest":
        return type(self)(
            prepared_payload=self.prepared_payload,
            initials=self.initials,
            t_span=self.t_span,
            solver_config=self.solver_config,
            simulation_identity=self.simulation_identity,
            parameter_overrides=self.parameter_overrides,
            intervention_schedule=intervention_schedule,
            version=self.version,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SimulationExecutionRequest":
        if "schedule_parameter_values" in payload:
            raise SimulationPreparationError(
                "execution_request",
                "stale schedule_parameter_values payload is not accepted; use parameter_overrides or committed/source parameter values.",
            )
        t_span_raw = payload.get("t_span") or (0.0, 0.0)
        try:
            t_span = (float(t_span_raw[0]), float(t_span_raw[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise SimulationPreparationError("execution_request", f"Invalid execution request t_span: {exc}") from exc
        prepared_payload = payload.get("prepared_payload")
        if prepared_payload is not None and not isinstance(prepared_payload, Mapping):
            raise SimulationPreparationError("execution_request", "Execution request prepared_payload must be a mapping.")
        intervention_schedule_present = "intervention_schedule" in payload
        intervention_schedule = None
        if intervention_schedule_present and payload.get("intervention_schedule") is not None:
            try:
                coerced_schedule = coerce_intervention_schedule(payload.get("intervention_schedule"))
            except InterventionScheduleError as exc:
                raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
            intervention_schedule = payload.get("intervention_schedule") if coerced_schedule is None else coerced_schedule
        kwargs: Dict[str, Any] = dict(
            version=int(payload.get("version") or 1),
            prepared_payload=dict(prepared_payload) if isinstance(prepared_payload, Mapping) else None,
            initials={str(name): float(value) for name, value in dict(payload.get("initials") or {}).items()},
            t_span=t_span,
            solver_config=dict(payload.get("solver_config") or {}),
            simulation_identity=(
                dict(payload.get("simulation_identity") or {})
                if isinstance(payload.get("simulation_identity"), Mapping)
                else None
            ),
            parameter_overrides=(
                {
                    str(name): float(value)
                    for name, value in dict(payload.get("parameter_overrides") or {}).items()
                }
                if isinstance(payload.get("parameter_overrides"), Mapping)
                else None
            ),
        )
        if intervention_schedule_present:
            kwargs["intervention_schedule"] = intervention_schedule
        return cls(**kwargs)

    def to_payload(self) -> Dict[str, Any]:
        prepared_payload: Optional[Dict[str, Any]] = None
        if isinstance(self.prepared_payload, Mapping):
            prepared_payload = dict(self.prepared_payload)
            if "y0" in prepared_payload:
                prepared_payload["y0"] = np.array(prepared_payload["y0"], copy=True, dtype=float).reshape(-1)
        payload = {
            "version": int(self.version),
            "prepared_payload": prepared_payload,
            "initials": {str(name): float(value) for name, value in dict(self.initials or {}).items()},
            "t_span": (float(self.t_span[0]), float(self.t_span[1])),
            "solver_config": dict(self.solver_config or {}),
            "simulation_identity": dict(self.simulation_identity or {}) if self.simulation_identity else None,
        }
        if self.parameter_overrides:
            payload["parameter_overrides"] = {
                str(name): float(value)
                for name, value in dict(self.parameter_overrides or {}).items()
            }
        if self.has_intervention_schedule_authority:
            if self.intervention_schedule is None:
                payload["intervention_schedule"] = None
                return payload
            try:
                schedule = coerce_intervention_schedule(self.intervention_schedule)
            except InterventionScheduleError as exc:
                raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
            payload["intervention_schedule"] = schedule.to_payload() if schedule is not None else {}
        return payload


_MISSING = object()


def coerce_simulation_execution_request(
    value: SimulationExecutionRequest | Mapping[str, Any] | None,
) -> Optional[SimulationExecutionRequest]:
    if value is None:
        return None
    if isinstance(value, SimulationExecutionRequest):
        return value
    if isinstance(value, Mapping):
        return SimulationExecutionRequest.from_mapping(value)
    raise SimulationPreparationError("execution_request", "Execution request must be a mapping.")


def _execution_request_schedule_payload_for_identity(
    request_payload: Mapping[str, Any],
) -> object:
    if "intervention_schedule" in request_payload:
        return request_payload.get("intervention_schedule")
    return None


def _execution_request_schedule_identity_mechanism(
    request: SimulationExecutionRequest,
) -> object:
    prepared_payload = request.prepared_payload
    if isinstance(prepared_payload, Mapping):
        return _mechanism_from_prepared_payload(prepared_payload)
    raise SimulationPreparationError(
        "intervention_schedule",
        "Execution request schedule identity validation requires a prepared mechanism payload.",
    )


def assert_simulation_execution_request_schedule_identity(
    request: SimulationExecutionRequest,
    *,
    expected_declarative_fingerprint: str,
    expected_executable_fingerprint: str = "",
) -> tuple[str, str]:
    mechanism = _execution_request_schedule_identity_mechanism(request)
    request_payload = request.to_payload()
    request_schedule = _execution_request_schedule_payload_for_identity(request_payload)
    mechanism_namespace = build_namespace_from_mechanism(mechanism)
    request_normalized_payload = normalized_intervention_schedule_payload(
        coerce_intervention_schedule(request_schedule),
        mechanism_namespace=mechanism_namespace,
    )
    if request_normalized_payload is None:
        request_declarative_fingerprint = ""
        request_executable_fingerprint = ""
    else:
        (
            request_declarative_fingerprint,
            request_executable_fingerprint,
        ) = intervention_schedule_identity_fingerprints(
            InterventionSchedule.from_payload(request_normalized_payload)
        )
    expected_declarative = str(expected_declarative_fingerprint or "")
    if expected_declarative != request_declarative_fingerprint:
        raise SimulationPreparationError(
            "intervention_schedule",
            "prepared_metadata.intervention_schedule_declarative_fingerprint conflicts with execution request.",
        )
    if str(expected_executable_fingerprint or "") != request_executable_fingerprint:
        raise SimulationPreparationError(
            "intervention_schedule",
            "prepared_metadata.intervention_schedule_executable_fingerprint conflicts with execution request.",
        )
    prepared_payload = request_payload.get("prepared_payload")
    if not isinstance(prepared_payload, Mapping):
        return request_declarative_fingerprint, request_executable_fingerprint
    for field_name in ("intervention_schedule", "unresolved_intervention_schedule"):
        if field_name not in prepared_payload:
            continue
        prepared_normalized_payload = normalized_intervention_schedule_payload(
            coerce_intervention_schedule(prepared_payload.get(field_name)),
            mechanism_namespace=mechanism_namespace,
        )
        if prepared_normalized_payload != request_normalized_payload:
            raise SimulationPreparationError(
                "intervention_schedule",
                (
                    f"prepared_payload.{field_name} conflicts with execution request "
                    "intervention schedule."
                ),
            )
    return request_declarative_fingerprint, request_executable_fingerprint


def _schedule_authority_decision_for_request(
    request_payload: SimulationExecutionRequest | None,
) -> _ScheduleAuthorityDecision:
    if request_payload is None or not request_payload.has_intervention_schedule_authority:
        return _ScheduleAuthorityDecision(_ScheduleAuthorityState.ABSENT)
    if request_payload.intervention_schedule is None:
        return _ScheduleAuthorityDecision(_ScheduleAuthorityState.EXPLICIT_NONE)
    try:
        schedule = coerce_intervention_schedule(request_payload.intervention_schedule)
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
    if schedule is None:
        return _ScheduleAuthorityDecision(_ScheduleAuthorityState.EXPLICIT_NONE)
    return _ScheduleAuthorityDecision(
        _ScheduleAuthorityState.EXPLICIT_SCHEDULE,
        unresolved_schedule=schedule,
    )


def _fresh_preparation_unresolved_intervention_schedule_for_request(
    request_payload: SimulationExecutionRequest,
) -> InterventionSchedule | None:
    decision = _schedule_authority_decision_for_request(request_payload)
    if decision.state is _ScheduleAuthorityState.EXPLICIT_SCHEDULE:
        return decision.unresolved_schedule
    return None


def _resolve_intervention_schedule_for_request(
    schedule: InterventionSchedule | None,
    request_payload: SimulationExecutionRequest | None,
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
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc


def materialize_request_intervention_schedule_for_parameter_values(
    *,
    mechanism: object,
    request: object,
    unresolved_intervention_schedule: InterventionSchedule | None,
    parameter_values: Mapping[str, Any],
    species_names: Iterable[str],
    runtime_parameter_names: Iterable[str] | None = None,
) -> object:
    if unresolved_intervention_schedule is None:
        return request
    parameter_partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides=parameter_values,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        runtime_parameter_names=runtime_parameter_names,
    )
    try:
        _raise_unowned_request_parameter_values(parameter_partition)
    except ValueError as exc:
        raise SimulationPreparationError("parameter_binding", str(exc)) from exc
    try:
        resolved_schedule = _resolve_intervention_schedule_for_request(
            unresolved_intervention_schedule,
            request if isinstance(request, SimulationExecutionRequest) else None,
            parameter_values=_schedule_resolution_values_for_request(
                request if isinstance(request, SimulationExecutionRequest) else None,
                unresolved_intervention_schedule,
                parameter_partition,
                mechanism=mechanism,
            ),
        )
        if resolved_schedule is not None:
            resolved_schedule.validate_species(species_names)
    except SimulationPreparationError:
        raise
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
    if isinstance(request, SimulationExecutionRequest):
        return request.with_intervention_schedule(resolved_schedule)
    if isinstance(request, SimulationRequest):
        return replace(
            request,
            intervention_execution=compile_intervention_schedule(
                resolved_schedule,
                cancellation_check=_cancellation_check_from_events(
                    request.events
                ),
            ),
        )
    raise SimulationPreparationError(
        "intervention_schedule",
        "Prepared fitting request does not support intervention schedule replacement.",
    )


def _mechanism_for_execution_request_schedule_materialization(
    request_payload: SimulationExecutionRequest,
) -> Any:
    prepared_payload = request_payload.prepared_payload
    if isinstance(prepared_payload, Mapping):
        return _mechanism_from_prepared_payload(prepared_payload)
    raise SimulationPreparationError(
        "prepared_payload",
        "Schedule materialization requires retained typed prepared payload.",
    )


def _schedule_parameters_deferred_to_prepared_request(
    request_payload: SimulationExecutionRequest | None,
    unresolved_intervention_schedule: InterventionSchedule | None,
) -> frozenset[str]:
    if request_payload is None or unresolved_intervention_schedule is None:
        return frozenset()
    if not unresolved_intervention_schedule.is_parameterized:
        return frozenset()
    prepared_payload = request_payload.prepared_payload
    if not isinstance(prepared_payload, Mapping):
        return frozenset()
    requested = prepared_payload.get("requested_parameter_names")
    if not isinstance(requested, (list, tuple, set, frozenset)):
        return frozenset()
    supplied = set(dict(request_payload.parameter_overrides or {}).keys())
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    return frozenset(
        str(name)
        for name in requested
        if str(name).strip()
        and str(name) in referenced
        and str(name) not in supplied
    )


def materialized_intervention_schedule_for_execution_request(
    request_payload: SimulationExecutionRequest | Mapping[str, Any],
) -> InterventionSchedule | None:
    request = coerce_simulation_execution_request(request_payload)
    if request is None:
        return None
    unresolved_intervention_schedule = _fresh_preparation_unresolved_intervention_schedule_for_request(
        request
    )
    if unresolved_intervention_schedule is None:
        return None
    if _schedule_parameters_deferred_to_prepared_request(
        request,
        unresolved_intervention_schedule,
    ):
        return None
    try:
        mechanism = _mechanism_for_execution_request_schedule_materialization(request)
        runtime_parameter_names, runtime_bindings = _install_requested_energy_runtime_bindings(
            mechanism,
            request.parameter_overrides,
        )
        parameter_partition = partition_simulation_parameter_values(
            mechanism=mechanism,
            parameter_overrides=request.parameter_overrides,
            unresolved_intervention_schedule=unresolved_intervention_schedule,
            runtime_parameter_names=runtime_parameter_names,
        )
        if request.parameter_overrides:
            if parameter_partition.unbound_mechanism_parameter_names:
                _bind_parameters_to_mechanism(
                    mechanism,
                    sorted(parameter_partition.unbound_mechanism_parameter_names),
                )
                parameter_partition = partition_simulation_parameter_values(
                    mechanism=mechanism,
                    parameter_overrides=request.parameter_overrides,
                    unresolved_intervention_schedule=unresolved_intervention_schedule,
                    runtime_parameter_names=runtime_parameter_names,
                )
            _raise_unowned_request_parameter_values(parameter_partition)
            _apply_parameter_overrides_to_prepared_mechanism(
                mechanism,
                parameter_partition=parameter_partition,
                runtime_bindings=runtime_bindings,
            )
        return _resolve_intervention_schedule_for_request(
            unresolved_intervention_schedule,
            request,
            parameter_values=_schedule_resolution_values_for_request(
                request,
                unresolved_intervention_schedule,
                parameter_partition,
                mechanism=mechanism,
            ),
        )
    except SimulationPreparationError:
        raise
    except ValueError as exc:
        raise SimulationPreparationError("parameter_binding", str(exc)) from exc
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
    except Exception as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc


def intervention_schedule_identity_for_execution_request(
    request_payload: SimulationExecutionRequest | Mapping[str, Any],
) -> tuple[str, str]:
    materialized_schedule = materialized_intervention_schedule_for_execution_request(
        request_payload
    )
    if materialized_schedule is None:
        return "", ""
    return intervention_schedule_identity_fingerprints(materialized_schedule)


def coerce_prepared_simulation_metadata(
    prepared: object,
) -> Optional[PreparedSimulationMetadata]:
    if prepared is None:
        return None
    if isinstance(prepared, PreparedSimulationMetadata):
        return prepared
    if isinstance(prepared, Mapping):
        return PreparedSimulationMetadata.from_mapping(prepared)
    return None


@dataclass(frozen=True, slots=True)
class _PreparedSolverConfig:
    solver_input: str
    solver: str
    rtol: float
    atol: float
    grid: Mapping[str, Any]
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool


@dataclass(frozen=True, slots=True)
class _PreparedRunContext:
    solver_config: _PreparedSolverConfig
    temperature_schedule: TemperatureScheduleProtocol | None
    symbolic_jacobian: SymbolicJacobianExecution
    algebra_text: Optional[str]
    warnings: Tuple[str, ...]

    @property
    def jacobian_func(self) -> Any:
        return self.symbolic_jacobian.jacobian_func

    @property
    def jac_sparsity(self) -> Any:
        return self.symbolic_jacobian.jac_sparsity

    @property
    def symbolic_jacobian_identity(self) -> Optional[Dict[str, Any]]:
        return dict(self.symbolic_jacobian.identity) if self.symbolic_jacobian.identity else None

    @property
    def symbolic_jacobian_status(self) -> Optional[Dict[str, Any]]:
        return dict(self.symbolic_jacobian.status) if self.symbolic_jacobian.status else None


def _build_solver_config(
    *,
    solver_input: str,
    rtol: object,
    atol: object,
    grid: object,
    use_sparse_jacobian: bool,
    wegscheider_cyclicity_enabled: bool,
) -> _PreparedSolverConfig:
    solver_label = str(solver_input or DEFAULT_SOLVER_NAME).strip() or DEFAULT_SOLVER_NAME
    solver = normalize_solver_name(solver_label)

    rtol_f = float(rtol)
    atol_f = float(atol)
    if not (np.isfinite(rtol_f) and rtol_f > 0.0):
        raise ValueError("rtol must be finite and > 0.")
    if not (np.isfinite(atol_f) and atol_f > 0.0):
        raise ValueError("atol must be finite and > 0.")

    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("grid must be {'dt': float} or {'N': int}")
    grid_out: Mapping[str, Any] = normalize_time_grid_policy(grid)

    return _PreparedSolverConfig(
        solver_input=str(solver_label),
        solver=str(solver),
        rtol=float(rtol_f),
        atol=float(atol_f),
        grid=grid_out,
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )


_REQUIRED_INTERNAL_SOLVER_CONFIG_KEYS = frozenset(
    {
        "solver",
        "rtol",
        "atol",
        "grid",
        MechanismMetadataKeys.TEMPERATURE_K,
        "use_sparse_jacobian",
        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
    }
)


def require_complete_internal_solver_config(
    solver_config: Mapping[str, Any] | None,
    *,
    stage: str = "solver_config",
) -> Dict[str, Any]:
    if not isinstance(solver_config, Mapping):
        raise SimulationPreparationError(stage, "Internal execution requires solver_config.")
    solver_config_out = dict(solver_config)
    missing = sorted(
        key
        for key in _REQUIRED_INTERNAL_SOLVER_CONFIG_KEYS
        if key not in solver_config_out
    )
    if missing:
        raise SimulationPreparationError(
            stage,
            "Internal execution solver_config is missing required keys: "
            + ", ".join(missing),
        )
    grid = solver_config_out.get("grid")
    if not isinstance(grid, Mapping) or not grid:
        raise SimulationPreparationError(stage, "Internal execution solver_config.grid is required.")
    try:
        solver_config_out["grid"] = normalize_time_grid_policy(grid)
    except Exception as exc:
        raise SimulationPreparationError(stage, str(exc)) from exc
    return solver_config_out


def _metadata_view_for_mechanism(
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


def metadata_view_for_mechanism(
    mechanism: Any,
    *,
    temperature_schedule_override: object = _MISSING,
) -> MechanismMetadataView:
    return _metadata_view_for_mechanism(
        mechanism,
        temperature_schedule_override=temperature_schedule_override,
    )


def _prepare_preparation_failure(stage: str, message: object) -> SimulationPreparationError:
    return SimulationPreparationError(str(stage or "unknown"), str(message or ""))


def _prepared_payload_failure(message: object) -> SimulationPreparationError:
    return _prepare_preparation_failure("prepared_payload", message)


def _clone_prepared_authority_mechanism(mechanism: Any) -> Any:
    clone_method = getattr(mechanism, "clone", None)
    if callable(clone_method):
        return clone_method()
    return copy.deepcopy(mechanism)


def _mechanism_from_prepared_payload(
    prepared_payload: Mapping[str, Any],
) -> Any:
    version = prepared_payload.get("version")
    if type(version) is not int:
        raise _prepared_payload_failure(
            "Prepared payload version must be an exact int."
        )
    if version != 4:
        raise _prepared_payload_failure(
            f"Unsupported prepared payload version: {version}"
        )
    try:
        mechanism_blob = prepared_payload["mechanism_blob"]
    except KeyError as exc:
        raise _prepared_payload_failure(
            "Missing prepared payload field: mechanism_blob"
        ) from exc
    if type(mechanism_blob) is not bytes:
        raise _prepared_payload_failure(
            "Prepared payload mechanism_blob must be exact bytes."
        )
    try:
        return pickle.loads(mechanism_blob)
    except Exception as exc:
        raise _prepared_payload_failure(
            f"Prepared mechanism payload is invalid: {exc}"
        ) from exc


def _preparation_failure_payload(exc: SimulationPreparationError) -> Dict[str, Any]:
    from kindred.core.simulation_failure import build_simulation_failure

    return build_simulation_failure(
        "preparation_error",
        str(exc),
        details={"stage": str(exc.stage or "unknown")},
        exc_type=exc.__class__.__name__,
    )


def _fit_simulation_error_from_preparation_error(
    exc: SimulationPreparationError,
    *,
    fatal: bool = True,
) -> FitSimulationError:
    return FitSimulationError(
        str(exc),
        details={
            "fatal": bool(fatal),
            "failure": _preparation_failure_payload(exc),
        },
    )


def _validated_prepared_worker_payload(
    prepared_payload: Mapping[str, Any],
) -> tuple[
    Any,
    Callable[..., np.ndarray] | None,
    list[str],
    np.ndarray,
    object,
    InterventionSchedule | None,
    InterventionSchedule | None,
    Any,
]:
    mechanism = _mechanism_from_prepared_payload(prepared_payload)
    rhs = None

    species_source = prepared_payload.get("species_names")
    if species_source is None:
        try:
            species_source = getattr(mechanism, "species_names")()
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Unable to determine species_names from prepared payload: {exc}"
            ) from exc
    try:
        species_names = [str(name) for name in species_source]
    except Exception as exc:
        raise _prepared_payload_failure(f"Invalid prepared payload species_names: {exc}") from exc
    if not species_names:
        raise _prepared_payload_failure("Prepared payload species_names must not be empty")

    y0_source = prepared_payload.get("y0")
    if y0_source is None:
        try:
            y0 = np.array(
                [mechanism.species[sp].initial_conc for sp in species_names],
                dtype=float,
            ).reshape(-1)
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Unable to derive prepared payload initial concentrations: {exc}"
            ) from exc
    else:
        try:
            y0 = np.array(y0_source, copy=True, dtype=float).reshape(-1)
        except Exception as exc:
            raise _prepared_payload_failure(f"Invalid prepared payload y0: {exc}") from exc
    if y0.size != len(species_names):
        raise _prepared_payload_failure(
            "Prepared payload y0 length does not match species_names length"
        )

    temperature_schedule_override: object = _MISSING
    if "temperature_schedule" in prepared_payload:
        temperature_schedule_override = prepared_payload.get("temperature_schedule")
        try:
            coerce_temperature_schedule(temperature_schedule_override)
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Invalid prepared payload temperature_schedule: {exc}"
            ) from exc
    intervention_schedule_override = None
    if "intervention_schedule" in prepared_payload:
        try:
            intervention_schedule_override = coerce_intervention_schedule(
                prepared_payload.get("intervention_schedule")
            )
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Invalid prepared payload intervention_schedule: {exc}"
            ) from exc
    unresolved_intervention_schedule_override = None
    if "unresolved_intervention_schedule" in prepared_payload:
        try:
            unresolved_intervention_schedule_override = coerce_intervention_schedule(
                prepared_payload.get("unresolved_intervention_schedule")
            )
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Invalid prepared payload unresolved_intervention_schedule: {exc}"
            ) from exc
    jacobian_func_override = prepared_payload.get("jacobian_func")
    return (
        mechanism,
        rhs,
        species_names,
        y0,
        temperature_schedule_override,
        intervention_schedule_override,
        unresolved_intervention_schedule_override,
        jacobian_func_override,
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
            ) from exc
        if bool(require_finite) and not np.isfinite(value):
            raise SimulationPreparationError(
                "parameter_overrides",
                f"Parameter override {name!r} must be finite.",
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


def _scalar_parameter_is_derived(mechanism: Any, name: str) -> bool:
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
    request_payload: SimulationExecutionRequest | None,
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
        if _scalar_parameter_is_derived(mechanism, name):
            return False
        setter = getattr(scalar_bindings.get(name), "set", None)
        if callable(setter):
            return True
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping) and name in scalar_info:
        return not _scalar_parameter_is_derived(mechanism, name)
    scalar_params = meta.get("scalar_params")
    if isinstance(scalar_params, dict) and name in scalar_params:
        if _scalar_parameter_is_derived(mechanism, name):
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
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

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
    parameter_partition: SimulationParameterValuePartition,
    runtime_bindings: Mapping[str, Any] | None = None,
) -> _ParameterOverrideApplication:
    """Apply slider-style values and report whether dependent runtime math changed."""
    partition = parameter_partition
    _raise_unowned_request_parameter_values(partition)
    if not partition.raw_values:
        return _ParameterOverrideApplication(rebuild_rhs=False)

    runtime_names = set(partition.runtime_parameter_names)
    runtime_binding_by_name = {
        str(name): binding
        for name, binding in dict(runtime_bindings or {}).items()
        if str(name)
    }
    runtime_override_items = sorted(
        (str(name), float(value))
        for name, value in partition.mechanism_binding_values.items()
        if str(name) in runtime_names
    )
    mechanism_override_items = sorted(
        (str(name), float(value))
        for name, value in partition.mechanism_binding_values.items()
        if str(name) not in runtime_names
    )
    if not mechanism_override_items and not runtime_override_items:
        return _ParameterOverrideApplication(rebuild_rhs=False)

    from kindred.core.simulator.step_indexing import lookup_step_param_target

    override_values_by_name = {str(name): float(value) for name, value in mechanism_override_items}
    _reject_conflicting_equilibrium_reverse_overrides(mechanism, override_values_by_name)
    binding_target_names: set[str] = set()
    for name, _value in mechanism_override_items:
        target_name = _canonical_step_override_name(mechanism, name)
        target = lookup_step_param_target(mechanism, target_name)
        if target is None:
            continue
        kind, idx, role, entry = target
        if kind == "equilibrium":
            _target_role, binding_name = _equilibrium_override_binding_target(
                mechanism,
                idx=int(idx),
                role=str(role),
                entry=entry,
            )
            binding_target_names.add(binding_name)
    if binding_target_names:
        _bind_parameters_to_mechanism(mechanism, sorted(binding_target_names))

    internal_names = _internal_parameter_algebra_binding_names(
        mechanism,
        requested_names=[name for name, _value in mechanism_override_items],
    )
    internal_names = {
        name
        for name in internal_names
        if not _prepared_parameter_override_can_apply(mechanism, name)
    }
    if internal_names:
        _bind_parameters_to_mechanism(mechanism, sorted(internal_names))

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
            target_role, _target_name, target_value = _equilibrium_override_assignment(
                mechanism,
                idx=int(idx),
                role=str(role),
                entry=entry,
                value=float(value),
                override_values_by_name=override_values_by_name,
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
            parameter_algebra_spec_from_mechanism,
        )

        spec = parameter_algebra_spec_from_mechanism(mechanism)
        if spec is not None:
            apply_parameter_algebra_spec_to_mechanism(
                spec,
                mechanism=mechanism,
                require_mutable=True,
            )
    for name, value in runtime_override_items:
        binding = runtime_binding_by_name.get(str(name))
        setter = getattr(binding, "set", None)
        if not callable(setter):
            raise SimulationPreparationError(
                "parameter_overrides",
                f"Runtime parameter {name!r} has no prepared execution binding.",
            )
        setter(float(value))
        override_applied = True
    return _ParameterOverrideApplication(
        rebuild_rhs=bool(override_applied),
    )


def apply_parameter_overrides_to_prepared_mechanism(
    mechanism: Any,
    *,
    parameter_partition: SimulationParameterValuePartition,
    runtime_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Apply slider-style values and report whether dependent runtime math changed."""
    return bool(
        _apply_parameter_overrides_to_prepared_mechanism(
            mechanism,
            parameter_partition=parameter_partition,
            runtime_bindings=runtime_bindings,
        ).rebuild_rhs
    )


def _canonical_step_override_name(mechanism: Any, name: str) -> str:
    name_s = str(name or "").strip()
    if not name_s:
        return ""
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

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
        )
    return str(role), _equilibrium_step_parameter_name(entry, str(role))


def _equilibrium_override_assignment(
    mechanism: Any,
    *,
    idx: int,
    role: str,
    entry: Mapping[str, Any],
    value: float,
    override_values_by_name: Mapping[str, float],
) -> tuple[str, str, float]:
    target_role, target_name = _equilibrium_override_binding_target(
        mechanism,
        idx=int(idx),
        role=str(role),
        entry=entry,
    )
    return target_role, target_name, float(value)


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
            )
        keq_value = float(requested[keq_name])
        if keq_value <= 0.0:
            raise SimulationPreparationError(
                "parameter_overrides",
                f"equilibrium parameter {keq_name} must be positive.",
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
    if _scalar_parameter_is_derived(mechanism, name):
        raise SimulationPreparationError(
            "parameter_overrides",
            f"derived scalar parameter {name!r} cannot be used as direct runtime override authority.",
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


def _build_prepared_run_context(
    *,
    mechanism: Any,
    solver_config: _PreparedSolverConfig,
    temperature_schedule_override: object = _MISSING,
    jacobian_func_override: Any = None,
) -> _PreparedRunContext:
    mech_meta = _metadata_view_for_mechanism(
        mechanism,
        temperature_schedule_override=temperature_schedule_override,
    )
    temperature_schedule = mech_meta.temperature_schedule
    warnings: List[str] = []

    symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
        jacobian_func=jacobian_func_override,
        jac_sparsity=None,
        status=None,
    )

    if jacobian_func_override is not None and not symbolic_jacobian.has_executable_jacobian:
        message = (
            "Ignored non-symbolic Jacobian callable from prepared payload; "
            "using generated symbolic Jacobian or solver default Jacobian handling."
        )
        logger.warning("%s", message)
        warnings.append(message)

    implicit_jacobian_solver = str(solver_config.solver).upper() in {"RADAU", "BDF"}
    analytical_jacobian_preparation_required = (
        not symbolic_jacobian.has_executable_jacobian
        and implicit_jacobian_solver
    )
    if analytical_jacobian_preparation_required:
        try:
            jacobian_func, symbolic_jacobian_identity = _bind_symbolic_jacobian_for_current_mechanism(
                mechanism=mechanism,
                prepared_solver_config=solver_config,
                temperature_schedule=temperature_schedule,
            )
            symbolic_jacobian = SymbolicJacobianExecution.supported(
                jacobian_func=jacobian_func,
                identity=symbolic_jacobian_identity,
            )
        except UnsupportedSymbolicExpressionError as exc:
            symbolic_jacobian = _symbolic_jacobian_for_bind_failure(mechanism, exc)
            reason_text = str((symbolic_jacobian.status or {}).get("reason") or exc)
            symbolic_message = f"Symbolic Jacobian unsupported; using solver default Jacobian handling: {reason_text}"
            logger.warning("%s", symbolic_message)
            warnings.append(symbolic_message)

    return _PreparedRunContext(
        solver_config=solver_config,
        temperature_schedule=temperature_schedule,
        symbolic_jacobian=symbolic_jacobian,
        algebra_text=getattr(mech_meta, "algebra_text", None),
        warnings=tuple(warnings),
    )


def _apply_execution_parameter_algebra_and_cyclicity(
    *,
    mechanism: Any,
    prepared_solver_config: _PreparedSolverConfig,
    require_mutable: bool,
) -> None:
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
        parameter_algebra_spec_from_mechanism,
    )
    from kindred.core.simulator.wegscheider_symbolic import (
        UnresolvedWegscheiderCyclicityError,
        validate_wegscheider_cyclicity_resolved,
    )

    try:
        spec = parameter_algebra_spec_from_mechanism(mechanism)
        if spec is not None:
            _ = apply_parameter_algebra_spec_to_mechanism(
                spec,
                mechanism=mechanism,
                require_mutable=bool(require_mutable),
            )
    except UnresolvedWegscheiderCyclicityError as exc:
        raise SimulationPreparationError("wegscheider_cyclicity", str(exc)) from exc
    except Exception as exc:
        raise SimulationPreparationError("parameter_algebra", str(exc)) from exc

    if prepared_solver_config.wegscheider_cyclicity_enabled and bool(
        getattr(mechanism, "equilibria", []) or []
    ):
        try:
            report = validate_wegscheider_cyclicity_resolved(mechanism)
        except UnresolvedWegscheiderCyclicityError as exc:
            raise SimulationPreparationError("wegscheider_cyclicity", str(exc)) from exc
        except Exception as exc:
            raise SimulationPreparationError("parameter_algebra", str(exc)) from exc
        if report.cycles:
            meta = getattr(mechanism, "metadata", None)
            if isinstance(meta, dict):
                meta["symbolic_wegscheider_identity"] = dict(report.symbolic_identity)


def _symbolic_jacobian_structure_for_mechanism(
    *,
    mechanism: Any,
):
    from kindred.core.symbolic.jacobian import (
        build_symbolic_jacobian_structure,
        symbolic_jacobian_structure_fingerprint_for_mechanism,
    )

    try:
        structure_fingerprint = symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism)
    except UnsupportedSymbolicExpressionError:
        return None
    key = symbolic_jacobian_structure_cache_key(
        structure_fingerprint=structure_fingerprint,
    )

    def _build_structure() -> Any:
        return build_symbolic_jacobian_structure(mechanism)

    try:
        return get_or_build_symbolic_jacobian_structure(key, _build_structure)
    except UnsupportedSymbolicExpressionError:
        return None


def _symbolic_jacobian_status_for_mechanism(mechanism: Any) -> Dict[str, Any]:
    from kindred.core.symbolic.jacobian import classify_symbolic_jacobian_support

    try:
        support = classify_symbolic_jacobian_support(mechanism)
    except UnsupportedSymbolicExpressionError as exc:
        return symbolic_status_payload(
            kind="jacobian",
            state="unsupported",
            code="classification-failed",
            reason=str(exc),
        )
    return dict(support.to_status_payload())


def _symbolic_jacobian_for_bind_failure(
    mechanism: Any,
    exc: UnsupportedSymbolicExpressionError,
) -> SymbolicJacobianExecution:
    classified_status = _symbolic_jacobian_status_for_mechanism(mechanism)
    return SymbolicJacobianExecution.from_bind_failure(
        classified_status=classified_status,
        exc=exc,
    )


def _prepared_metadata_with_symbolic_jacobian(
    metadata: PreparedSimulationMetadata,
    symbolic_jacobian: SymbolicJacobianExecution,
) -> PreparedSimulationMetadata:
    return replace(metadata, **symbolic_jacobian.metadata_kwargs())


def _bind_symbolic_jacobian_for_current_mechanism(
    *,
    mechanism: Any,
    prepared_solver_config: _PreparedSolverConfig,
    temperature_schedule: TemperatureScheduleProtocol | None,
) -> tuple[Any, Dict[str, Any]]:
    if str(prepared_solver_config.solver).upper() not in {"BDF", "RADAU"}:
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires BDF or Radau."
        )
    structure = _symbolic_jacobian_structure_for_mechanism(
        mechanism=mechanism,
    )
    if structure is None:
        raise UnsupportedSymbolicExpressionError("symbolic structure is unsupported.")
    from kindred.core.ode_builder import (
        build_effective_rate_coefficient_evaluator,
    )

    schedule_identity = None
    if temperature_schedule is not None:
        if not isinstance(
            temperature_schedule,
            TemperatureScheduleDictProtocol,
        ):
            raise UnsupportedSymbolicExpressionError(
                "Symbolic Jacobian requires a deterministic typed "
                "temperature schedule; arbitrary callables are unsupported."
            )
        schedule_identity = dict(
            temperature_schedule_execution_identity(temperature_schedule)
            or {}
        )
    coefficient_evaluator = build_effective_rate_coefficient_evaluator(
        mechanism
    )
    symbolic_artifact = structure.bind(
        coefficient_evaluator,
        temperature_schedule=temperature_schedule,
        schedule_identity=schedule_identity,
    )
    return symbolic_artifact.jacobian_func, symbolic_artifact.identity.to_payload()


def prepare_simulation_worker_run(
    *,
    execution_request: SimulationExecutionRequest | Mapping[str, Any],
    events: Optional[Iterable[Callable[[float, np.ndarray], float]]] = None,
    progress_callback: Optional[Callable[[float, float, float], None]] = None,
) -> PreparedSimulationRun:
    """
    Prepare an execution-ready SimulationRequest for the GUI SimulationWorker.

    This function is the single owner for:
    - Temperature schedule coercion (DSL schedule takes precedence)
    - Parameter algebra application
    - Metadata flags (e.g. Wegscheider cyclicity enablement)
    - Generated symbolic Jacobian construction when supported by the solver path
    """

    event_values = tuple(events or ())
    request_payload = coerce_simulation_execution_request(execution_request)
    if request_payload is None:
        raise SimulationPreparationError(
            "execution_request",
            "Worker preparation requires an execution request.",
        )
    initials = dict(request_payload.initials or {})
    t_span = (
        float(request_payload.t_span[0]),
        float(request_payload.t_span[1]),
    )
    prepared_payload = request_payload.prepared_payload
    if not isinstance(prepared_payload, Mapping):
        raise SimulationPreparationError(
            "prepared_payload",
            "Worker preparation requires retained typed prepared payload.",
        )
    solver_config = require_complete_internal_solver_config(
        request_payload.solver_config,
        stage="execution_request.solver_config",
    )
    solver_config = require_complete_internal_solver_config(
        solver_config,
        stage="solver_config",
    )

    try:
        prepared_solver_config = _build_solver_config(
            solver_input=str(solver_config["solver"]),
            rtol=solver_config["rtol"],
            atol=solver_config["atol"],
            grid=solver_config["grid"],
            use_sparse_jacobian=bool(solver_config["use_sparse_jacobian"]),
            wegscheider_cyclicity_enabled=bool(
                solver_config[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED]
            ),
        )
    except Exception as exc:
        raise SimulationPreparationError("solver_config", str(exc)) from exc

    mechanism: Any
    rhs: Callable[..., np.ndarray] | None
    species_names: List[str]
    y0: np.ndarray
    temperature_schedule_override: object = _MISSING
    intervention_schedule_override: InterventionSchedule | None = None
    unresolved_intervention_schedule_override: InterventionSchedule | None = None
    jacobian_func_override = None

    try:
        (
            mechanism,
            rhs,
            species_names,
            y0,
            temperature_schedule_override,
            intervention_schedule_override,
            unresolved_intervention_schedule_override,
            jacobian_func_override,
        ) = _validated_prepared_worker_payload(prepared_payload)

        for idx, sp in enumerate(species_names):
            if sp in initials:
                y0[idx] = float(initials[sp])
    except Exception as exc:
        if isinstance(exc, SimulationPreparationError):
            raise
        raise _prepared_payload_failure(
            f"Invalid prepared payload: {exc}"
        ) from exc
    require_mutable = True

    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            prepared_solver_config.wegscheider_cyclicity_enabled
        )

    _apply_execution_parameter_algebra_and_cyclicity(
        mechanism=mechanism,
        prepared_solver_config=prepared_solver_config,
        require_mutable=bool(require_mutable),
    )

    unresolved_intervention_schedule = (
        _fresh_preparation_unresolved_intervention_schedule_for_request(request_payload)
        if request_payload is not None
        else unresolved_intervention_schedule_override
    )
    parameter_partition: SimulationParameterValuePartition | None = None
    rebuild_rhs_for_parameter_overrides = False
    if request_payload is not None and (
        bool(request_payload.parameter_overrides)
        or bool(
            unresolved_intervention_schedule is not None
            and unresolved_intervention_schedule.is_parameterized
        )
    ):
        try:
            runtime_parameter_names = _available_energy_binding_names(mechanism)
            parameter_partition = partition_simulation_parameter_values(
                mechanism=mechanism,
                parameter_overrides=request_payload.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
                runtime_parameter_names=runtime_parameter_names,
            )
            mechanism_override_names = sorted(parameter_partition.bindable_mechanism_parameter_names)
            internal_names = _internal_parameter_algebra_binding_names(
                mechanism,
                requested_names=mechanism_override_names,
            )
            _bind_parameters_to_mechanism(
                mechanism,
                sorted(set(mechanism_override_names) | set(internal_names)),
            )
            runtime_parameter_names, runtime_bindings = _install_requested_energy_runtime_bindings(
                mechanism,
                request_payload.parameter_overrides,
            )
            parameter_partition = partition_simulation_parameter_values(
                mechanism=mechanism,
                parameter_overrides=request_payload.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
                runtime_parameter_names=runtime_parameter_names,
            )
            rebuild_rhs_for_parameter_overrides = apply_parameter_overrides_to_prepared_mechanism(
                mechanism,
                parameter_partition=parameter_partition,
                runtime_bindings=runtime_bindings,
            )
        except Exception as exc:
            raise SimulationPreparationError("parameter_overrides", str(exc)) from exc

    intervention_schedule = None
    try:
        if request_payload is not None:
            intervention_schedule = unresolved_intervention_schedule
        elif intervention_schedule_override is not None:
            intervention_schedule = intervention_schedule_override
        else:
            meta_schedule = (getattr(mechanism, "metadata", {}) or {}).get(
                MechanismMetadataKeys.INTERVENTION_SCHEDULE
            )
            intervention_schedule = coerce_intervention_schedule(meta_schedule)
        deferred_schedule_parameters = _schedule_parameters_deferred_to_prepared_request(
            request_payload,
            intervention_schedule,
        )
        if deferred_schedule_parameters:
            intervention_schedule.validate_species(species_names)
            intervention_schedule = None
        else:
            intervention_schedule = _resolve_intervention_schedule_for_request(
                intervention_schedule,
                request_payload,
                parameter_values=_schedule_resolution_values_for_request(
                    request_payload,
                    unresolved_intervention_schedule,
                    parameter_partition,
                    mechanism=mechanism,
                ),
            )
        if intervention_schedule is not None:
            intervention_schedule.validate_species(species_names)
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc

    if rhs is None or bool(rebuild_rhs_for_parameter_overrides):
        try:
            from kindred.core.ode_builder import build_ode_rhs_from_mechanism

            rhs = build_ode_rhs_from_mechanism(mechanism)
        except Exception as exc:
            raise SimulationPreparationError("ode_build", str(exc)) from exc

    try:
        prepared_context = _build_prepared_run_context(
            mechanism=mechanism,
            solver_config=prepared_solver_config,
            temperature_schedule_override=temperature_schedule_override,
            jacobian_func_override=jacobian_func_override,
        )
    except Exception as exc:
        raise SimulationPreparationError("temperature_schedule", str(exc)) from exc

    temperature_schedule = prepared_context.temperature_schedule
    symbolic_wegscheider_identity = None
    meta = getattr(mechanism, "metadata", {}) or {}
    if isinstance(meta, Mapping) and isinstance(meta.get("symbolic_wegscheider_identity"), Mapping):
        symbolic_wegscheider_identity = dict(meta.get("symbolic_wegscheider_identity") or {})

    try:
        initials_for_algebra = {sp: float(y0[idx]) for idx, sp in enumerate(species_names)}
    except Exception as exc:
        raise SimulationPreparationError("initials_for_algebra", str(exc)) from exc

    intervention_execution = compile_intervention_schedule(
        intervention_schedule,
        cancellation_check=_cancellation_check_from_events(event_values),
    )
    runtime_symbolic_jacobian = prepared_context.symbolic_jacobian
    if (
        str(prepared_solver_config.solver).upper() in {"BDF", "RADAU"}
        and not prepared_solver_config.use_sparse_jacobian
    ):
        runtime_symbolic_jacobian = SymbolicJacobianExecution.disabled(
            code="policy-disabled",
            reason="Symbolic Jacobian runtime use disabled by sparse Jacobian policy.",
        )
    request = SimulationRequest(
        rhs=rhs,
        t_span=(float(t_span[0]), float(t_span[1])),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        intervention_execution=intervention_execution,
        solver=str(prepared_solver_config.solver),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        grid=prepared_solver_config.grid,
        **runtime_symbolic_jacobian.to_request_kwargs(),
        events=list(event_values) if events is not None else None,
        temperature_schedule=temperature_schedule,
        species_names=tuple(species_names),
        progress_callback=progress_callback,
        symbolic_wegscheider_identity=symbolic_wegscheider_identity,
    )

    return PreparedSimulationRun(
        mechanism=mechanism,
        rhs=rhs,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        species_names=list(species_names),
        solver_input=str(prepared_solver_config.solver_input),
        temperature_schedule=temperature_schedule,
        intervention_execution=intervention_execution,
        jacobian_func=prepared_context.jacobian_func,
        jac_sparsity=prepared_context.jac_sparsity,
        initials_for_algebra=initials_for_algebra,
        warnings=list(prepared_context.warnings),
        request=request,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )


def _bind_parameters_to_mechanism(mech: Any, names: List[str]) -> Dict[str, Any]:
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
                    default_kappa=float(mechanism_metadata.kappa_global),
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
                        default_kappa=float(
                            mechanism_metadata.kappa_global
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

    return bindings


def _internal_parameter_algebra_binding_names(
    mechanism: Any,
    requested_names: Iterable[str] = (),
) -> set[str]:
    from kindred.core.equilibrium_rate_authority import authority_fields_from_step_entry, step_entry_authored_role_is_editable
    from kindred.core.simulator.parameter_algebra import parameter_algebra_spec_from_mechanism
    from kindred.core.simulator.step_indexing import get_step_index_map

    try:
        spec = parameter_algebra_spec_from_mechanism(mechanism)
    except Exception as exc:
        raise SimulationPreparationError(
            "parameter_algebra",
            f"Failed to inspect parameter algebra bindings: {exc}",
        ) from exc
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
    active_keq_names = set(constrained_keq_targets)
    requested_canonical = {
        _canonical_step_override_name(mechanism, str(name))
        for name in (requested_names or ())
        if str(name or "").strip()
    }
    try:
        step_entries = list(get_step_index_map(mechanism))
    except Exception as exc:
        raise SimulationPreparationError(
            "parameter_algebra",
            f"Failed to inspect step-index parameter ownership: {exc}",
        ) from exc
    for entry in step_entries:
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        step_idx_raw = entry.get("step_index")
        if isinstance(step_idx_raw, int):
            n = int(step_idx_raw)
        elif isinstance(step_idx_raw, str) and step_idx_raw.isdigit():
            n = int(step_idx_raw)
        else:
            continue
        keq_name = f"Keq{n}"
        authority = authority_fields_from_step_entry(entry)
        has_keq_param = bool(step_entry_authored_role_is_editable(entry, "Keq"))
        if not authority:
            raise SimulationPreparationError(
                "parameter_algebra",
                f"Equilibrium step {n} is missing normalized equilibrium_authority metadata.",
            )
        if has_keq_param and (
            keq_name in requested_canonical
            or f"kf{n}" in requested_canonical
            or f"kr{n}" in requested_canonical
        ):
            active_keq_names.add(keq_name)
    return constrained - constrained_keq_targets


def _reject_requested_algebra_owned_mechanism_parameters_for_fitting(
    mechanism: Any,
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
        for name in (requested & _internal_parameter_algebra_binding_names(mechanism))
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


def _available_energy_binding_names(mechanism: Any) -> frozenset[str]:
    _ = mechanism
    return frozenset()


def _install_requested_energy_runtime_bindings(
    mechanism: Any,
    parameter_overrides: Mapping[str, Any] | None,
) -> tuple[frozenset[str], Dict[str, Any]]:
    runtime_names = _available_energy_binding_names(mechanism)
    if not runtime_names or not isinstance(parameter_overrides, Mapping):
        return runtime_names, {}
    requested = {
        str(name or "").strip()
        for name in parameter_overrides.keys()
        if str(name or "").strip()
    }
    requested_runtime_names = sorted(str(name) for name in runtime_names if str(name) in requested)
    if not requested_runtime_names:
        return runtime_names, {}
    return runtime_names, _install_energy_bindings(mechanism, requested_runtime_names)


def _install_energy_bindings(mechanism: Any, names: List[str]) -> Dict[str, Any]:
    _ = (mechanism, names)
    return {}


def _bind_requested_parameters_to_prepared_mechanism(
    mechanism: Any,
    requested_names: Iterable[str],
    *,
    unresolved_intervention_schedule: InterventionSchedule | None = None,
) -> Dict[str, Any]:
    requested = [str(name) for name in (requested_names or ()) if str(name).strip()]
    parameter_partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides=None,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        requested_parameter_names=requested,
        runtime_parameter_names=_available_energy_binding_names(mechanism),
    )
    _raise_unowned_request_parameter_values(parameter_partition)
    mechanism_names = set(parameter_partition.bindable_mechanism_parameter_names)
    mechanism_names.update(
        _internal_parameter_algebra_binding_names(
            mechanism,
            requested_names=mechanism_names,
        )
    )
    bindings = _bind_parameters_to_mechanism(mechanism, sorted(mechanism_names))
    energy_bindings = _install_energy_bindings(
        mechanism,
        sorted(parameter_partition.runtime_parameter_names),
    )
    scalar_bindings = (getattr(mechanism, "metadata", {}) or {}).get("scalar_param_bindings") or {}
    if isinstance(scalar_bindings, dict):
        for name, binding in scalar_bindings.items():
            if str(name) in set(parameter_partition.scalar_parameter_names):
                bindings.setdefault(str(name), binding)
    bindings.update(energy_bindings)
    return bindings


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
    from kindred.core.ode_builder import build_ode_rhs_from_mechanism

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    mechanism = prepared_dsl.mechanism_for_execution()
    unresolved_intervention_schedule = coerce_intervention_schedule(
        prepared_dsl.intervention_schedule_for_execution()
    )
    spec = prepared_dsl.parameter_algebra_spec_for_execution()
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            wegscheider_cyclicity_enabled
        )
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

    try:
        from kindred.core.simulator.parameter_algebra import (
            mechanism_parameter_namespace,
        )

        mechanism_namespace = mechanism_parameter_namespace(mechanism)
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
            runtime_parameter_names=_available_energy_binding_names(mechanism),
        )
        _raise_unowned_request_parameter_values(
            requested_parameter_partition,
            allow_unbound_mechanism_parameters=True,
        )
        canonical_requested_param_names = canonicalize_request_parameter_names(
            requested_parameter_partition,
            param_names or [],
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
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parameter_algebra", exc)
        ) from exc

    bindings = _bind_parameters_to_mechanism(mechanism, mech_bind_names)
    try:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )
        from kindred.core.simulator.wegscheider_symbolic import UnresolvedWegscheiderCyclicityError

        _ = apply_parameter_algebra_spec_to_mechanism(
            spec,
            mechanism=mechanism,
            require_mutable=True,
        )
    except UnresolvedWegscheiderCyclicityError as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("wegscheider_cyclicity", exc)
        ) from exc
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parameter_algebra", exc)
        ) from exc

    energy_bindings = _install_energy_bindings(mechanism, list(param_names or []))

    scalar_bindings = (getattr(mechanism, "metadata", {}) or {}).get("scalar_param_bindings") or {}
    if isinstance(scalar_bindings, dict):
        public_scalar_binding_names = set(requested_parameter_partition.scalar_parameter_names)
        for name, binding in scalar_bindings.items():
            if str(name) in public_scalar_binding_names:
                bindings.setdefault(str(name), binding)
    bindings.update(energy_bindings)
    public_param_names = sorted(
        {
            str(name)
            for name in canonical_requested_param_names
            if str(name).strip() and str(name) in bindings
        }
    )

    try:
        rhs = build_ode_rhs_from_mechanism(mechanism)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("ode_build", exc)
        ) from exc

    bound = BoundMechanism(
        mechanism=mechanism,
        rhs=rhs,
        bindings=bindings,
        species_names=species_names,
        y0=y0,
        param_names=public_param_names,
        dsl_analysis=prepared_dsl.analysis,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )

    def rebuild_bound_rhs() -> None:
        bound.rhs = build_ode_rhs_from_mechanism(bound.mechanism)

    for binding in energy_bindings.values():
        register = getattr(binding, "add_after_set_callback", None)
        if callable(register):
            register(rebuild_bound_rhs)

    return bound
