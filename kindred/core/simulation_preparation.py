"""
Simulation preparation utilities used by fitting and prepared/bound execution paths.

This module intentionally owns the "parse DSL → bind parameters → compile RHS" pipeline
so that optimization code and analysis code can depend on a narrower surface area.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict

import numpy as np

from kindred.core.rate_binding import RateBinding
import kindred.core.simulator.solvers as solver_api
from kindred.core.exceptions import FitSimulationError
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.simulation_series_payload import SimulationSeriesPayload
from kindred.core.temperature import TemperatureScheduleProtocol, coerce_temperature_schedule
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    InterventionScheduleError,
    coerce_intervention_schedule,
    intervention_schedule_parameter_names,
    normalized_intervention_schedule_fingerprint,
    normalized_intervention_schedule_payload,
    parse_intervention_schedule_from_dsl,
)
from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.simulator.solvers import (
    DEFAULT_SOLVER_NAME,
    SimulationRequest,
    normalize_solver_name,
    solve_ode,
)
from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.identity import normalize_symbolic_identity_mapping
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.symbolic.namespaces import symbolic_status_payload
from kindred.core.symbolic.structure_cache import (
    clear_symbolic_jacobian_structure_cache,
    get_or_build_symbolic_jacobian_structure,
    symbolic_jacobian_structure_cache_key,
    symbolic_jacobian_structure_cache_stats,
)
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoundMechanism",
    "SimulationExecutionRequest",
    "PreparedSimulationMetadata",
    "PreparedFittingObjectiveContext",
    "PreparedSimulationRun",
    "SimulationParameterValuePartition",
    "SimulationPreparationError",
    "assert_simulation_execution_request_schedule_identity",
    "build_simulation_request_from_prepared_run",
    "build_prepared_simulation_func",
    "coerce_prepared_simulation_metadata",
    "metadata_view_for_mechanism",
    "prepared_simulation_run_for_execution_request",
    "prepare_fitting_objective_context",
    "prepare_bound_mechanism",
    "prepare_simulation_worker_run",
    "materialize_request_intervention_schedule_for_parameter_values",
    "partition_simulation_parameter_values",
    "canonicalize_request_parameter_names",
    "clear_symbolic_jacobian_structure_cache",
    "symbolic_jacobian_identity_for_execution_text",
    "symbolic_jacobian_structure_cache_stats",
    "symbolic_wegscheider_identity_for_execution_text",
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
    mechanism_text: str
    unresolved_intervention_schedule: InterventionSchedule | None = None

    def as_execution_payload(
        self,
        *,
        include_rhs: bool,
    ) -> "SimulationWorkerPreparedPayloadV1 | SimulationExecutionPreparedPayloadV2":
        """Return a structured execution payload for worker or batch execution."""
        metadata = metadata_view_for_mechanism(self.mechanism)
        intervention_schedule = getattr(metadata, "intervention_schedule", None)
        payload: dict[str, Any] = {
            "version": 2,
            "mechanism": self.mechanism,
            "y0": np.array(self.y0, copy=True),
            "species_names": list(self.species_names),
            "mechanism_text": self.mechanism_text,
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
        if include_rhs:
            payload["version"] = 1
            payload["rhs"] = self.rhs
        return payload

    def as_worker_payload(self) -> "SimulationWorkerPreparedPayloadV1":
        """Return a payload suitable for SimulationWorker prepared mode."""
        return self.as_execution_payload(include_rhs=True)  # type: ignore[return-value]

    def as_serializable_execution_payload(self) -> "SimulationExecutionPreparedPayloadV2":
        """Return a process-safe prepared payload without the unpicklable RHS closure."""
        return self.as_execution_payload(include_rhs=False)  # type: ignore[return-value]


class SimulationPreparationError(RuntimeError):
    """
    Error raised when preparing a SimulationWorker run payload/request fails.

    The GUI worker maps `stage` values to user-facing messages while keeping the
    preparation logic in core.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(str(message))
        self.stage = str(stage or "unknown")


def _solve_request(request: SimulationRequest):
    # Resolve through the solver module so callers that monkeypatch
    # `kindred.core.simulator.solvers.solve_ode` keep working even if this
    # module was imported before the patch was applied.
    solver = getattr(solver_api, solve_ode.__name__, solve_ode)
    return solver(request)


@dataclass(frozen=True)
class PreparedSimulationRun:
    """Execution-ready simulation payload produced by core preparation utilities."""

    mechanism: Any
    rhs: Callable[..., np.ndarray]
    y0: np.ndarray
    species_names: List[str]
    solver_input: str
    solver_warning: Optional[str]
    temperature_schedule: TemperatureScheduleProtocol | None
    intervention_schedule: InterventionSchedule | None
    jacobian_func: Any
    jac_sparsity: Any
    initials_for_algebra: Optional[Dict[str, float]]
    warnings: List[str]
    request: SimulationRequest | SimulationExecutionRequest
    unresolved_intervention_schedule: InterventionSchedule | None = None


def resolve_prepared_run_intervention_schedule(
    prepared_run: PreparedSimulationRun,
    parameter_partition: SimulationParameterValuePartition,
) -> InterventionSchedule | None:
    request = prepared_run.request
    if not isinstance(request, SimulationRequest):
        raise SimulationPreparationError(
            "intervention_schedule",
            "Prepared run intervention schedule resolution requires a solver SimulationRequest.",
        )
    intervention_schedule = request.intervention_schedule
    if intervention_schedule is None:
        return None
    try:
        return intervention_schedule.resolve_parameters(
            parameter_partition.schedule_resolution_values
        )
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc


def build_simulation_request_from_prepared_run(
    prepared_run: PreparedSimulationRun,
    *,
    y0: np.ndarray,
    intervention_schedule: InterventionSchedule | None,
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
        solver=str(request.solver),
        rtol=float(request.rtol),
        atol=float(request.atol),
        grid=dict(request.grid or {}),
        **symbolic_jacobian.to_request_kwargs(),
        temperature_schedule=request.temperature_schedule,
        intervention_schedule=intervention_schedule,
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
    solver_warning: Optional[str]
    rtol: float
    atol: float
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool
    initial_prefix: str
    intervention_schedule_fingerprint: str = ""
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
            "solver_warning": str(self.solver_warning) if self.solver_warning else None,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "use_sparse_jacobian": bool(self.use_sparse_jacobian),
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: bool(
                self.wegscheider_cyclicity_enabled
            ),
            "initial_prefix": str(self.initial_prefix),
            "intervention_schedule_fingerprint": str(self.intervention_schedule_fingerprint or ""),
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
        param_names_raw = meta.get("param_names") or []
        if not isinstance(param_names_raw, (list, tuple)):
            param_names_raw = []
        solver_requested = str(
            meta.get("solver_requested")
            or meta.get("solver_input")
            or meta.get("solver")
            or ""
        ).strip()
        solver_normalized = str(
            meta.get("solver_normalized")
            or meta.get("solver_used")
            or meta.get("solver")
            or ""
        ).strip()
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
            solver_warning=str(meta.get("solver_warning")) if meta.get("solver_warning") else None,
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
            intervention_schedule_fingerprint=str(meta.get("intervention_schedule_fingerprint") or ""),
            symbolic_jacobian_identity=meta.get("symbolic_jacobian_identity"),
            symbolic_jacobian_status=(
                dict(meta.get("symbolic_jacobian_status") or {})
                if isinstance(meta.get("symbolic_jacobian_status"), Mapping)
                else None
            ),
            symbolic_wegscheider_identity=meta.get("symbolic_wegscheider_identity"),
        )


@dataclass(frozen=True)
class PreparedFittingObjectiveContext:
    """Prepared context for building a fitting residual objective."""

    bound: BoundMechanism
    requested_param_names: List[str]
    request: Any
    target_species: str
    target_is_species: bool
    target_species_index: Optional[int]
    compiled_algebra: Any
    initials_for_algebra: Dict[str, float]
    temperature_K: float
    unresolved_intervention_schedule: InterventionSchedule | None = None
    warnings: List[str] = field(default_factory=list)


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
    mechanism_text: str = ""
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
        mechanism_text: str = "",
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
        object.__setattr__(self, "mechanism_text", str(mechanism_text or ""))
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
            mechanism_text=self.mechanism_text,
            simulation_identity=self.simulation_identity,
            parameter_overrides=self.parameter_overrides,
            intervention_schedule=intervention_schedule,
            version=self.version,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SimulationExecutionRequest":
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
            mechanism_text=str(payload.get("mechanism_text") or ""),
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
            "mechanism_text": str(self.mechanism_text or ""),
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


class SimulationWorkerPreparedPayloadV1(TypedDict):
    version: int
    mechanism: Any
    rhs: Callable[..., np.ndarray]
    y0: np.ndarray
    species_names: List[str]
    mechanism_text: str
    temperature_schedule: TemperatureScheduleProtocol | None
    intervention_schedule: Mapping[str, Any] | None
    unresolved_intervention_schedule: Mapping[str, Any] | None
    jacobian_func: Any


class SimulationExecutionPreparedPayloadV2(TypedDict):
    version: int
    mechanism: Any
    y0: np.ndarray
    species_names: List[str]
    mechanism_text: str
    temperature_schedule: TemperatureScheduleProtocol | None
    intervention_schedule: Mapping[str, Any] | None
    unresolved_intervention_schedule: Mapping[str, Any] | None
    jacobian_func: Any


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
        mechanism = prepared_payload.get("mechanism")
        if mechanism is not None:
            return mechanism
    raise SimulationPreparationError(
        "intervention_schedule",
        "Execution request schedule identity validation requires a prepared mechanism payload.",
    )


def _normalized_schedule_payload_for_mechanism(
    schedule: object,
    *,
    mechanism: object,
) -> dict[str, Any] | None:
    return normalized_intervention_schedule_payload(
        coerce_intervention_schedule(schedule),
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
    )


def _normalized_schedule_fingerprint_for_mechanism(
    schedule: object,
    *,
    mechanism: object,
) -> str:
    return normalized_intervention_schedule_fingerprint(
        coerce_intervention_schedule(schedule),
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
    )


def assert_simulation_execution_request_schedule_identity(
    request: SimulationExecutionRequest,
    *,
    expected_fingerprint: str,
) -> str:
    mechanism = _execution_request_schedule_identity_mechanism(request)
    request_payload = request.to_payload()
    request_schedule = _execution_request_schedule_payload_for_identity(request_payload)
    request_normalized_payload = _normalized_schedule_payload_for_mechanism(
        request_schedule,
        mechanism=mechanism,
    )
    request_fingerprint = _normalized_schedule_fingerprint_for_mechanism(
        request_schedule,
        mechanism=mechanism,
    )
    if str(expected_fingerprint or "") != request_fingerprint:
        raise SimulationPreparationError(
            "intervention_schedule",
            "prepared_metadata.intervention_schedule_fingerprint conflicts with execution request.",
        )
    prepared_payload = request_payload.get("prepared_payload")
    if not isinstance(prepared_payload, Mapping):
        return request_fingerprint
    for field_name in ("intervention_schedule", "unresolved_intervention_schedule"):
        if field_name not in prepared_payload:
            continue
        prepared_normalized_payload = _normalized_schedule_payload_for_mechanism(
            prepared_payload.get(field_name),
            mechanism=mechanism,
        )
        if prepared_normalized_payload != request_normalized_payload:
            raise SimulationPreparationError(
                "intervention_schedule",
                (
                    f"prepared_payload.{field_name} conflicts with execution request "
                    "intervention schedule."
                ),
            )
    return request_fingerprint


def _execution_request_intervention_schedule(
    request_payload: SimulationExecutionRequest,
) -> InterventionSchedule | None:
    try:
        if request_payload.has_intervention_schedule_authority:
            if request_payload.intervention_schedule is None:
                return None
            schedule = coerce_intervention_schedule(request_payload.intervention_schedule)
            return schedule
        return None
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc


def _unresolved_intervention_schedule_for_request(
    request_payload: SimulationExecutionRequest | None,
) -> InterventionSchedule | None:
    decision = _schedule_authority_decision_for_request(request_payload)
    if decision.state is not _ScheduleAuthorityState.EXPLICIT_SCHEDULE:
        return None
    return decision.unresolved_schedule


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


def _prepared_reuse_unresolved_intervention_schedule_for_request(
    request_payload: SimulationExecutionRequest | None,
    *,
    prepared_unresolved_intervention_schedule: InterventionSchedule | None = None,
) -> InterventionSchedule | None:
    decision = _schedule_authority_decision_for_request(request_payload)
    if decision.state is _ScheduleAuthorityState.ABSENT:
        return prepared_unresolved_intervention_schedule
    if decision.state is _ScheduleAuthorityState.EXPLICIT_NONE:
        return None
    return decision.unresolved_schedule


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
    allow_deferred_parameters: bool = False,
    parameter_values: Mapping[str, Any] | None = None,
) -> InterventionSchedule | None:
    if schedule is None:
        return None
    if parameter_values is None and schedule.is_parameterized:
        if bool(allow_deferred_parameters):
            return schedule
        raise SimulationPreparationError(
            "intervention_schedule",
            "Parameterized intervention schedules require typed request parameter classification before resolution.",
        )
    values = dict(parameter_values or {})
    if schedule.is_parameterized and not values:
        if bool(allow_deferred_parameters):
            return schedule
        try:
            schedule.resolve_parameters({})
        except InterventionScheduleError as exc:
            raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
        return schedule
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
            parameter_values=parameter_partition.schedule_resolution_values,
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
        return replace(request, intervention_schedule=resolved_schedule)
    raise SimulationPreparationError(
        "intervention_schedule",
        "Prepared fitting request does not support intervention schedule replacement.",
    )


def _prepared_request_allows_deferred_schedule_parameters(
    request_payload: SimulationExecutionRequest | None,
    *,
    structured_prepared_request: bool,
) -> bool:
    if request_payload is None or not bool(structured_prepared_request):
        return False
    prepared_payload = request_payload.prepared_payload
    if not isinstance(prepared_payload, Mapping):
        return False
    return isinstance(prepared_payload.get("bindings"), Mapping)


def coerce_prepared_simulation_metadata(
    prepared: object,
) -> Optional[PreparedSimulationMetadata]:
    if prepared is None:
        return None
    if isinstance(prepared, PreparedSimulationMetadata):
        return prepared
    if isinstance(prepared, Mapping):
        try:
            return PreparedSimulationMetadata.from_mapping(prepared)
        except Exception:
            return None
    return None


@dataclass(frozen=True, slots=True)
class _PreparedSolverConfig:
    solver_input: str
    solver: str
    solver_warning: Optional[str]
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
    solver, solver_warning = normalize_solver_name(solver_label)

    rtol_f = float(rtol)
    atol_f = float(atol)
    if not (np.isfinite(rtol_f) and rtol_f > 0.0):
        raise ValueError("rtol must be finite and > 0.")
    if not (np.isfinite(atol_f) and atol_f > 0.0):
        raise ValueError("atol must be finite and > 0.")

    if not isinstance(grid, Mapping) or not grid:
        grid_out: Mapping[str, Any] = {"N": 100}
    else:
        grid_out = dict(grid)

    return _PreparedSolverConfig(
        solver_input=str(solver_label),
        solver=str(solver),
        solver_warning=str(solver_warning) if solver_warning else None,
        rtol=float(rtol_f),
        atol=float(atol_f),
        grid=grid_out,
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )


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


def _mechanism_has_dynamic_rate_bindings(mechanism: Any) -> bool:
    for reaction in getattr(mechanism, "reactions", []) or []:
        value = getattr(reaction, "rate", None)
        if isinstance(value, RateBinding) or callable(value):
            return True
    for equilibrium in getattr(mechanism, "equilibria", []) or []:
        for attr_name in ("kf", "kr", "Keq"):
            value = getattr(equilibrium, attr_name, None)
            if isinstance(value, RateBinding) or callable(value):
                return True
        meta = getattr(equilibrium, "metadata", {}) or {}
        if isinstance(meta, Mapping):
            keq_input = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
            if isinstance(keq_input, RateBinding) or callable(keq_input):
                return True
    return False


def _mechanism_has_dynamic_keq_input_binding(mechanism: Any) -> bool:
    for equilibrium in getattr(mechanism, "equilibria", []) or []:
        meta = getattr(equilibrium, "metadata", {}) or {}
        if not isinstance(meta, Mapping):
            continue
        keq_input = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        if isinstance(keq_input, RateBinding) or callable(keq_input):
            return True
    return False


def _mechanism_supports_dynamic_symbolic_snapshot(mechanism: Any) -> bool:
    return not _mechanism_has_dynamic_keq_input_binding(mechanism)


def _prepare_preparation_failure(stage: str, message: object) -> SimulationPreparationError:
    return SimulationPreparationError(str(stage or "unknown"), str(message or ""))


def _prepared_payload_failure(message: object) -> SimulationPreparationError:
    return _prepare_preparation_failure("prepared_payload", message)


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
) -> tuple[Any, Callable[..., np.ndarray] | None, list[str], np.ndarray, object, InterventionSchedule | None, Any]:
    try:
        version = int(prepared_payload.get("version", 1))
    except Exception as exc:
        raise _prepared_payload_failure(f"Invalid prepared payload version: {exc}") from exc
    if version not in {1, 2}:
        raise _prepared_payload_failure(
            f"Unsupported prepared payload version: {version}"
        )

    try:
        mechanism = prepared_payload["mechanism"]
    except KeyError as exc:
        raise _prepared_payload_failure(f"Missing prepared payload field: {exc.args[0]}") from exc
    if version == 1:
        try:
            rhs = prepared_payload["rhs"]
        except KeyError as exc:
            raise _prepared_payload_failure(f"Missing prepared payload field: {exc.args[0]}") from exc
    else:
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


def _scalar_parameter_override_known(mechanism: Any, name: str) -> bool:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return False
    scalar_bindings = meta.get("scalar_param_bindings")
    if isinstance(scalar_bindings, dict) and name in scalar_bindings:
        setter = getattr(scalar_bindings.get(name), "set", None)
        if callable(setter):
            return True
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping) and name in scalar_info:
        return True
    scalar_params = meta.get("scalar_params")
    if isinstance(scalar_params, dict) and name in scalar_params:
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
        from kindred.core.equilibrium_rate_authority import require_step_entry_role_editable

        try:
            require_step_entry_role_editable(entry, role, parameter_name=target_name)
        except ValueError as exc:
            raise SimulationPreparationError("parameter_overrides", str(exc)) from exc
    candidate = None
    try:
        if kind == "reaction" and role == "k":
            candidate = getattr(mechanism.reactions[int(idx)], "rate", None)
        elif kind == "equilibrium" and role in {"kf", "kr"}:
            candidate = getattr(mechanism.equilibria[int(idx)], str(role), None)
        elif kind == "equilibrium" and role == "Keq":
            meta = getattr(mechanism.equilibria[int(idx)], "metadata", {}) or {}
            if isinstance(meta, Mapping):
                candidate = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
    except Exception:
        candidate = None
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
) -> _ParameterOverrideApplication:
    """Apply slider-style values and report whether dependent runtime math changed."""
    partition = parameter_partition
    _raise_unowned_request_parameter_values(partition)
    if not partition.raw_values:
        return _ParameterOverrideApplication(rebuild_rhs=False)

    mechanism_override_items = sorted(partition.mechanism_binding_values.items())
    if not mechanism_override_items:
        return _ParameterOverrideApplication(rebuild_rhs=False)

    from kindred.core.simulator.step_indexing import lookup_step_param_target

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
        kind, idx, role, _entry = target
        candidate = None
        try:
            if kind == "reaction" and role == "k":
                candidate = getattr(mechanism.reactions[int(idx)], "rate", None)
            elif kind == "equilibrium" and role in {"kf", "kr"}:
                candidate = getattr(mechanism.equilibria[int(idx)], str(role), None)
            elif kind == "equilibrium" and role == "Keq":
                meta = getattr(mechanism.equilibria[int(idx)], "metadata", {}) or {}
                if isinstance(meta, Mapping):
                    candidate = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        except Exception:
            candidate = None

        setter = getattr(candidate, "set", None)
        if callable(setter):
            setter(float(value))
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
    return _ParameterOverrideApplication(
        rebuild_rhs=bool(override_applied),
    )


def apply_parameter_overrides_to_prepared_mechanism(
    mechanism: Any,
    *,
    parameter_partition: SimulationParameterValuePartition,
) -> bool:
    """Apply slider-style values and report whether dependent runtime math changed."""
    return bool(
        _apply_parameter_overrides_to_prepared_mechanism(
            mechanism,
            parameter_partition=parameter_partition,
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


def _apply_scalar_parameter_override_to_prepared_mechanism(
    mechanism: Any,
    name: str,
    value: float,
) -> bool:
    from kindred.core.rate_binding import RateBinding

    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return False
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


def prepared_simulation_run_for_execution_request(
    prepared: PreparedSimulationRun,
    execution_request: SimulationExecutionRequest | Mapping[str, Any],
) -> PreparedSimulationRun:
    """Reuse a prepared runtime while applying request-local initials and slider values."""
    request_payload = coerce_simulation_execution_request(execution_request)
    if request_payload is None:
        return prepared

    prepared_unresolved_schedule = (
        prepared.unresolved_intervention_schedule
        if prepared.unresolved_intervention_schedule is not None
        else prepared.request.intervention_schedule
    )
    unresolved_intervention_schedule = _prepared_reuse_unresolved_intervention_schedule_for_request(
        request_payload,
        prepared_unresolved_intervention_schedule=prepared_unresolved_schedule,
    )
    parameter_partition = partition_simulation_parameter_values(
        mechanism=prepared.mechanism,
        parameter_overrides=request_payload.parameter_overrides,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )
    try:
        _raise_unowned_request_parameter_values(
            parameter_partition,
            allow_unbound_mechanism_parameters=True,
        )
    except Exception as exc:
        raise SimulationPreparationError("parameter_overrides", str(exc)) from exc
    if request_payload.parameter_overrides and parameter_partition.unbound_mechanism_parameter_names:
        return prepare_simulation_worker_run(execution_request=request_payload)

    try:
        override_application = _apply_parameter_overrides_to_prepared_mechanism(
            prepared.mechanism,
            parameter_partition=parameter_partition,
        )
    except Exception as exc:
        raise SimulationPreparationError("parameter_overrides", str(exc)) from exc
    rebuild_rhs = bool(override_application.rebuild_rhs)
    rhs = prepared.request.rhs
    symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
        jacobian_func=prepared.request.jacobian_func,
        jac_sparsity=prepared.request.jac_sparsity,
        status=prepared.request.symbolic_jacobian_status,
    )
    temperature_schedule = prepared.request.temperature_schedule
    intervention_schedule = unresolved_intervention_schedule
    intervention_schedule = _resolve_intervention_schedule_for_request(
        intervention_schedule,
        request_payload,
        allow_deferred_parameters=True,
        parameter_values=parameter_partition.schedule_resolution_values,
    )
    warnings = list(getattr(prepared, "warnings", None) or [])
    if intervention_schedule is not None:
        try:
            intervention_schedule.validate_species(prepared.species_names)
        except InterventionScheduleError as exc:
            raise SimulationPreparationError("intervention_schedule", str(exc)) from exc
    if rebuild_rhs:
        try:
            from kindred.core.ode_builder import build_ode_rhs_from_mechanism

            rhs = build_ode_rhs_from_mechanism(prepared.mechanism)
        except Exception as exc:
            raise SimulationPreparationError("ode_build", str(exc)) from exc
        try:
            solver_config = _build_solver_config(
                solver_input=str(prepared.request.solver),
                rtol=float(prepared.request.rtol),
                atol=float(prepared.request.atol),
                grid=dict(prepared.request.grid or {}),
                use_sparse_jacobian=bool(prepared.request.jacobian_func is not None),
                wegscheider_cyclicity_enabled=bool(
                    (getattr(prepared.mechanism, "metadata", {}) or {}).get(
                        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                        WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                    )
                ),
            )
            prepared_context = _build_prepared_run_context(
                mechanism=prepared.mechanism,
                solver_config=solver_config,
                temperature_schedule_override=prepared.request.temperature_schedule,
                jacobian_func_override=None,
                allow_dynamic_binding_symbolic_snapshot=True,
            )
            symbolic_jacobian = prepared_context.symbolic_jacobian
            temperature_schedule = prepared_context.temperature_schedule
            warnings = list(prepared_context.warnings)
        except Exception as exc:
            raise SimulationPreparationError("parameter_overrides", str(exc)) from exc

    y0 = np.asarray(prepared.y0, dtype=float).reshape(-1).copy()
    initials = dict(request_payload.initials or {})
    if initials:
        for idx, sp in enumerate(prepared.species_names):
            if sp not in initials:
                continue
            try:
                y0[idx] = float(initials[sp])
            except (TypeError, ValueError, OverflowError):
                continue

    request = replace(
        prepared.request,
        rhs=rhs,
        t_span=(float(request_payload.t_span[0]), float(request_payload.t_span[1])),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        **symbolic_jacobian.to_request_kwargs(),
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
        species_names=tuple(prepared.species_names),
    )
    initials_for_algebra = {
        str(sp): float(y0[idx])
        for idx, sp in enumerate(prepared.species_names)
    }
    return replace(
        prepared,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        initials_for_algebra=initials_for_algebra,
        warnings=warnings,
        intervention_schedule=intervention_schedule,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        request=request,
    )


def _build_prepared_run_context(
    *,
    mechanism: Any,
    solver_config: _PreparedSolverConfig,
    temperature_schedule_override: object = _MISSING,
    jacobian_func_override: Any = None,
    allow_dynamic_binding_symbolic_snapshot: bool = False,
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
    if temperature_schedule is not None and (
        symbolic_jacobian.has_executable_jacobian
        or (solver_config.use_sparse_jacobian and implicit_jacobian_solver)
    ):
        message = "Symbolic Jacobian disabled for scheduled-temperature run; using solver default Jacobian handling."
        logger.warning("%s", message)
        warnings.append(message)
        symbolic_jacobian = SymbolicJacobianExecution.disabled(
            code="scheduled-temperature",
            reason="Symbolic Jacobian disabled for scheduled-temperature run.",
        )
    analytical_jacobian_requested = (
        not symbolic_jacobian.has_executable_jacobian
        and solver_config.use_sparse_jacobian
        and implicit_jacobian_solver
        and temperature_schedule is None
    )
    if (
        solver_config.use_sparse_jacobian
        and implicit_jacobian_solver
        and temperature_schedule is None
        and _mechanism_has_dynamic_rate_bindings(mechanism)
        and (
            not bool(allow_dynamic_binding_symbolic_snapshot)
            or not _mechanism_supports_dynamic_symbolic_snapshot(mechanism)
        )
    ):
        classified_status = _symbolic_jacobian_status_for_mechanism(mechanism)
        if classified_status.get("state") == "unsupported":
            symbolic_jacobian = SymbolicJacobianExecution.from_support_status(classified_status)
            reason_text = str(classified_status.get("reason") or "unsupported symbolic Jacobian")
            message = f"Symbolic Jacobian unsupported; using solver default Jacobian handling: {reason_text}"
        else:
            symbolic_jacobian = SymbolicJacobianExecution.unsupported(
                code="dynamic-rate-binding",
                reason="Symbolic Jacobian disabled for dynamic rate bindings.",
            )
            message = (
                "Symbolic Jacobian disabled for dynamic rate bindings; "
                "using solver default Jacobian handling."
            )
        logger.warning("%s", message)
        warnings.append(message)
    elif analytical_jacobian_requested:
        try:
            jacobian_func, symbolic_jacobian_identity = _bind_symbolic_jacobian_for_current_mechanism(
                mechanism=mechanism,
                prepared_solver_config=solver_config,
                temperature_K=float(mech_meta.temperature_K),
            )
            symbolic_jacobian = SymbolicJacobianExecution.supported(
                jacobian_func=jacobian_func,
                identity=symbolic_jacobian_identity,
            )
        except UnsupportedSymbolicExpressionError as exc:
            symbolic_jacobian = _symbolic_jacobian_for_bind_failure(mechanism, exc)
            reason_text = str((symbolic_jacobian.status or {}).get("reason") or exc)
            symbolic_message = f"Symbolic Jacobian unsupported; using solver default Jacobian handling: {reason_text}"
            logger.warning("%s", symbolic_message, exc_info=True)
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
    mechanism_text: str,
    structured_prepared_request: bool,
    prepared_solver_config: _PreparedSolverConfig,
    require_mutable: bool,
) -> None:
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
        apply_parameter_algebra_to_mechanism,
        parameter_algebra_spec_from_mechanism,
    )
    from kindred.core.simulator.wegscheider_symbolic import (
        UnresolvedWegscheiderCyclicityError,
        validate_wegscheider_cyclicity_resolved,
    )

    try:
        if structured_prepared_request:
            spec = parameter_algebra_spec_from_mechanism(mechanism)
            if spec is not None:
                _ = apply_parameter_algebra_spec_to_mechanism(
                    spec,
                    mechanism=mechanism,
                    require_mutable=bool(require_mutable),
                )
            elif prepared_solver_config.wegscheider_cyclicity_enabled:
                validate_wegscheider_cyclicity_resolved(mechanism)
        elif mechanism_text:
            _ = apply_parameter_algebra_to_mechanism(
                mechanism_text,
                mechanism=mechanism,
                require_mutable=bool(require_mutable),
            )
        else:
            spec = parameter_algebra_spec_from_mechanism(mechanism)
            if spec is not None:
                _ = apply_parameter_algebra_spec_to_mechanism(
                    spec,
                    mechanism=mechanism,
                    require_mutable=bool(require_mutable),
                )
            elif prepared_solver_config.wegscheider_cyclicity_enabled:
                validate_wegscheider_cyclicity_resolved(mechanism)
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


def symbolic_jacobian_identity_for_execution_text(
    *,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    solver_cfg = dict(solver_config or {})
    prepared_solver_config = _build_solver_config(
        solver_input=str(solver_cfg.get("solver") or DEFAULT_SOLVER_NAME),
        rtol=solver_cfg.get("rtol", 1e-6),
        atol=solver_cfg.get("atol", 1e-12),
        grid=solver_cfg.get("grid", {"N": 100}) or {"N": 100},
        use_sparse_jacobian=bool(
            solver_cfg.get("use_sparse_jacobian", USE_SPARSE_JACOBIAN_DEFAULT)
        ),
        wegscheider_cyclicity_enabled=bool(
            solver_cfg.get(
                MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
            )
        ),
    )
    if not prepared_solver_config.use_sparse_jacobian:
        return None
    if str(prepared_solver_config.solver or "").upper() not in {"BDF", "RADAU"}:
        return None
    temperature_K = float(solver_cfg.get(MechanismMetadataKeys.TEMPERATURE_K, 298.15))
    structure = _symbolic_jacobian_structure_cache_get(
        mechanism_text=str(mechanism_text or ""),
        prepared_solver_config=prepared_solver_config,
        temperature_K=temperature_K,
    )
    if structure is None:
        return None
    try:
        snapshot_values = _symbolic_jacobian_snapshot_values_for_execution_text(
            mechanism_text=str(mechanism_text or ""),
            prepared_solver_config=prepared_solver_config,
            temperature_K=temperature_K,
            parameter_overrides=parameter_overrides,
            parameter_symbols=tuple(getattr(structure, "parameter_symbols", ()) or ()),
        )
        return dict(structure.bind(snapshot_values).identity.to_payload())
    except UnsupportedSymbolicExpressionError:
        return None


def _parse_symbolic_execution_mechanism(
    *,
    mechanism_text: str,
    prepared_solver_config: _PreparedSolverConfig,
    temperature_K: float,
):
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.units import UnitsModel

    units = UnitsModel(temperature_K=float(temperature_K))
    mechanism = parse_dsl_to_mechanism(str(mechanism_text or ""), initials={}, units=units)
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            prepared_solver_config.wegscheider_cyclicity_enabled
        )
    _apply_execution_parameter_algebra_and_cyclicity(
        mechanism=mechanism,
        mechanism_text=str(mechanism_text or ""),
        structured_prepared_request=False,
        prepared_solver_config=_build_solver_config(
            solver_input=str(prepared_solver_config.solver),
            rtol=1e-6,
            atol=1e-12,
            grid={"N": 100},
            use_sparse_jacobian=True,
            wegscheider_cyclicity_enabled=bool(prepared_solver_config.wegscheider_cyclicity_enabled),
        ),
        require_mutable=False,
    )
    return mechanism


def _symbolic_jacobian_structure_for_mechanism(
    *,
    mechanism: Any,
    prepared_solver_config: _PreparedSolverConfig,
    temperature_K: float,
):
    from kindred.core.symbolic.jacobian import (
        build_symbolic_jacobian_structure,
        symbolic_jacobian_structure_fingerprint_for_mechanism,
    )

    if _metadata_view_for_mechanism(mechanism).temperature_schedule is not None:
        return None
    try:
        structure_fingerprint = symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism)
    except UnsupportedSymbolicExpressionError:
        return None
    key = symbolic_jacobian_structure_cache_key(
        structure_fingerprint=structure_fingerprint,
        solver=str(prepared_solver_config.solver),
        temperature_K=float(temperature_K),
        wegscheider_cyclicity_enabled=bool(prepared_solver_config.wegscheider_cyclicity_enabled),
    )

    def _build_structure() -> Any:
        return build_symbolic_jacobian_structure(mechanism)

    try:
        return get_or_build_symbolic_jacobian_structure(key, _build_structure)
    except UnsupportedSymbolicExpressionError:
        return None


def _symbolic_jacobian_structure_cache_get(
    *,
    mechanism_text: str,
    prepared_solver_config: _PreparedSolverConfig,
    temperature_K: float,
):
    mechanism = _parse_symbolic_execution_mechanism(
        mechanism_text=mechanism_text,
        prepared_solver_config=prepared_solver_config,
        temperature_K=temperature_K,
    )
    return _symbolic_jacobian_structure_for_mechanism(
        mechanism=mechanism,
        prepared_solver_config=prepared_solver_config,
        temperature_K=float(temperature_K),
    )


def _symbolic_jacobian_status_for_mechanism(mechanism: Any) -> Dict[str, Any]:
    from kindred.core.symbolic.jacobian import classify_symbolic_jacobian_support

    try:
        support = classify_symbolic_jacobian_support(mechanism)
    except Exception as exc:
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
    temperature_K: float,
) -> tuple[Any, Dict[str, Any]]:
    structure = _symbolic_jacobian_structure_for_mechanism(
        mechanism=mechanism,
        prepared_solver_config=prepared_solver_config,
        temperature_K=float(temperature_K),
    )
    if structure is None:
        raise UnsupportedSymbolicExpressionError("symbolic structure is unsupported.")
    snapshot_values = _symbolic_jacobian_parameter_values_from_mechanism(
        mechanism,
        parameter_symbols=tuple(getattr(structure, "parameter_symbols", ()) or ()),
    )
    symbolic_artifact = structure.bind(snapshot_values)
    return symbolic_artifact.jacobian_func, symbolic_artifact.identity.to_payload()


def _symbolic_jacobian_snapshot_values_for_execution_text(
    *,
    mechanism_text: str,
    prepared_solver_config: _PreparedSolverConfig,
    temperature_K: float,
    parameter_overrides: Mapping[str, Any] | None,
    parameter_symbols: tuple[str, ...],
) -> Dict[str, float]:
    mechanism = _parse_symbolic_execution_mechanism(
        mechanism_text=mechanism_text,
        prepared_solver_config=prepared_solver_config,
        temperature_K=temperature_K,
    )
    if parameter_overrides:
        unresolved_intervention_schedule = parse_intervention_schedule_from_dsl(str(mechanism_text or ""))
        parameter_partition = partition_simulation_parameter_values(
            mechanism=mechanism,
            parameter_overrides=parameter_overrides,
            unresolved_intervention_schedule=unresolved_intervention_schedule,
        )
        override_names = sorted(parameter_partition.bindable_mechanism_parameter_names)
        _reject_implicit_equilibrium_constant_overrides(mechanism, override_names)
        internal_names = _internal_parameter_algebra_binding_names(
            mechanism,
            requested_names=override_names,
        )
        _bind_parameters_to_mechanism(
            mechanism,
            sorted(set(override_names) | set(internal_names)),
        )
        parameter_partition = partition_simulation_parameter_values(
            mechanism=mechanism,
            parameter_overrides=parameter_overrides,
            unresolved_intervention_schedule=unresolved_intervention_schedule,
        )
        apply_parameter_overrides_to_prepared_mechanism(
            mechanism,
            parameter_partition=parameter_partition,
        )
    return _symbolic_jacobian_parameter_values_from_mechanism(
        mechanism,
        parameter_symbols=parameter_symbols,
    )


def _symbolic_jacobian_parameter_values_from_mechanism(
    mechanism: Any,
    *,
    parameter_symbols: tuple[str, ...],
) -> Dict[str, float]:
    from kindred.core.simulator.step_indexing import lookup_step_param_target
    from kindred.core.validation import try_parse_callable_finite_float

    values: Dict[str, float] = {}
    for name in parameter_symbols:
        target = lookup_step_param_target(mechanism, str(name))
        candidate = None
        if target is not None:
            kind, idx, role, _entry = target
            if kind == "reaction" and role == "k":
                candidate = getattr(mechanism.reactions[int(idx)], "rate", None)
            elif kind == "equilibrium" and role in {"kf", "kr"}:
                candidate = getattr(mechanism.equilibria[int(idx)], str(role), None)
            elif kind == "equilibrium" and role == "Keq":
                eq = mechanism.equilibria[int(idx)]
                meta = getattr(eq, "metadata", {}) or {}
                if isinstance(meta, Mapping):
                    candidate = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        parsed, ok = try_parse_callable_finite_float(candidate)
        if not ok:
            raise UnsupportedSymbolicExpressionError(f"Missing symbolic parameter value for {name!r}.")
        values[str(name)] = float(parsed)
    return values


def symbolic_wegscheider_identity_for_execution_text(
    *,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
) -> Optional[Dict[str, Any]]:
    solver_cfg = dict(solver_config or {})
    prepared_solver_config = _build_solver_config(
        solver_input=str(solver_cfg.get("solver") or DEFAULT_SOLVER_NAME),
        rtol=solver_cfg.get("rtol", 1e-6),
        atol=solver_cfg.get("atol", 1e-12),
        grid=solver_cfg.get("grid", {"N": 100}) or {"N": 100},
        use_sparse_jacobian=bool(
            solver_cfg.get("use_sparse_jacobian", USE_SPARSE_JACOBIAN_DEFAULT)
        ),
        wegscheider_cyclicity_enabled=bool(
            solver_cfg.get(
                MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
            )
        ),
    )
    if not prepared_solver_config.wegscheider_cyclicity_enabled:
        return None

    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.units import UnitsModel

    units = UnitsModel(temperature_K=float(solver_cfg.get(MechanismMetadataKeys.TEMPERATURE_K, 298.15)))
    mechanism = parse_dsl_to_mechanism(str(mechanism_text or ""), initials={}, units=units)
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = True
    _apply_execution_parameter_algebra_and_cyclicity(
        mechanism=mechanism,
        mechanism_text=str(mechanism_text or ""),
        structured_prepared_request=False,
        prepared_solver_config=prepared_solver_config,
        require_mutable=False,
    )
    meta = getattr(mechanism, "metadata", {}) or {}
    if isinstance(meta, Mapping) and isinstance(meta.get("symbolic_wegscheider_identity"), Mapping):
        return dict(meta.get("symbolic_wegscheider_identity") or {}) or None
    return None


def prepare_simulation_worker_run(
    *,
    mechanism_text: str = "",
    initials: Optional[Mapping[str, float]] = None,
    t_span: Tuple[float, float] = (0.0, 0.0),
    solver_config: Optional[Mapping[str, Any]] = None,
    prepared_payload: Optional[Mapping[str, Any]] = None,
    execution_request: SimulationExecutionRequest | Mapping[str, Any] | None = None,
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

    request_payload = coerce_simulation_execution_request(execution_request)
    structured_prepared_request = False
    if request_payload is not None:
        request_mechanism_text = str(request_payload.mechanism_text or "")
        initials = dict(request_payload.initials or {})
        t_span = (float(request_payload.t_span[0]), float(request_payload.t_span[1]))
        solver_config = dict(request_payload.solver_config or {})
        if prepared_payload is None:
            prepared_payload = request_payload.prepared_payload
        structured_prepared_request = prepared_payload is not None and request_payload.prepared_payload is not None
        if structured_prepared_request:
            mechanism_text = request_mechanism_text
        else:
            mechanism_text = str(request_mechanism_text or mechanism_text or "")

    if initials is None:
        initials = {}
    if solver_config is None:
        solver_config = {}

    try:
        prepared_solver_config = _build_solver_config(
            solver_input=str((solver_config or {}).get("solver") or DEFAULT_SOLVER_NAME),
            rtol=(solver_config or {}).get("rtol", 1e-6),
            atol=(solver_config or {}).get("atol", 1e-12),
            grid=(solver_config or {}).get("grid", {"N": 100}) or {"N": 100},
            use_sparse_jacobian=bool(
                (solver_config or {}).get(
                    "use_sparse_jacobian",
                    USE_SPARSE_JACOBIAN_DEFAULT,
                )
            ),
            wegscheider_cyclicity_enabled=bool(
                (solver_config or {}).get(
                    MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                )
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

    if prepared_payload is not None:
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

            if initials:
                for idx, sp in enumerate(species_names):
                    if sp in initials:
                        try:
                            y0[idx] = float(initials[sp])
                        except (TypeError, ValueError):
                            continue
        except Exception as exc:
            if isinstance(exc, SimulationPreparationError):
                raise
            raise _prepared_payload_failure(f"Invalid prepared payload: {exc}") from exc
        require_mutable = True
    else:
        from kindred.core.units import UnitsModel
        from kindred.core.simulator.dsl import parse_dsl_to_mechanism

        temperature_K = float(
            (solver_config or {}).get(MechanismMetadataKeys.TEMPERATURE_K, 298.15)
        )
        units = UnitsModel(temperature_K=temperature_K)
        try:
            mechanism = parse_dsl_to_mechanism(mechanism_text, initials=dict(initials or {}), units=units)
        except Exception as exc:
            raise SimulationPreparationError("parse", str(exc)) from exc

        species_names = list(getattr(mechanism, "species_names")())
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names], dtype=float)
        rhs = None
        require_mutable = False

    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            prepared_solver_config.wegscheider_cyclicity_enabled
        )

    _apply_execution_parameter_algebra_and_cyclicity(
        mechanism=mechanism,
        mechanism_text=str(mechanism_text or ""),
        structured_prepared_request=bool(structured_prepared_request),
        prepared_solver_config=prepared_solver_config,
        require_mutable=bool(require_mutable),
    )

    unresolved_intervention_schedule = (
        _fresh_preparation_unresolved_intervention_schedule_for_request(request_payload)
        if request_payload is not None
        else unresolved_intervention_schedule_override
    )
    parameter_partition: SimulationParameterValuePartition | None = None
    if request_payload is not None and (
        bool(request_payload.parameter_overrides)
        or bool(
            unresolved_intervention_schedule is not None
            and unresolved_intervention_schedule.is_parameterized
        )
    ):
        try:
            parameter_partition = partition_simulation_parameter_values(
                mechanism=mechanism,
                parameter_overrides=request_payload.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
            )
            mechanism_override_names = sorted(parameter_partition.bindable_mechanism_parameter_names)
            _reject_implicit_equilibrium_constant_overrides(mechanism, mechanism_override_names)
            internal_names = _internal_parameter_algebra_binding_names(
                mechanism,
                requested_names=mechanism_override_names,
            )
            _bind_parameters_to_mechanism(
                mechanism,
                sorted(set(mechanism_override_names) | set(internal_names)),
            )
            parameter_partition = partition_simulation_parameter_values(
                mechanism=mechanism,
                parameter_overrides=request_payload.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
            )
            apply_parameter_overrides_to_prepared_mechanism(
                mechanism,
                parameter_partition=parameter_partition,
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
        intervention_schedule = _resolve_intervention_schedule_for_request(
            intervention_schedule,
            request_payload,
            allow_deferred_parameters=_prepared_request_allows_deferred_schedule_parameters(
                request_payload,
                structured_prepared_request=structured_prepared_request,
            ),
            parameter_values=(
                parameter_partition.schedule_resolution_values
                if parameter_partition is not None
                else None
            ),
        )
        if intervention_schedule is not None:
            intervention_schedule.validate_species(species_names)
    except InterventionScheduleError as exc:
        raise SimulationPreparationError("intervention_schedule", str(exc)) from exc

    if rhs is None:
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
            allow_dynamic_binding_symbolic_snapshot=bool(
                request_payload is not None and request_payload.parameter_overrides
            ),
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

    request = SimulationRequest(
        rhs=rhs,
        t_span=(float(t_span[0]), float(t_span[1])),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        solver=str(prepared_solver_config.solver),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        grid=prepared_solver_config.grid,
        **prepared_context.symbolic_jacobian.to_request_kwargs(),
        events=list(events) if events is not None else None,
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
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
        solver_warning=str(prepared_solver_config.solver_warning) if prepared_solver_config.solver_warning else None,
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
        jacobian_func=prepared_context.jacobian_func,
        jac_sparsity=prepared_context.jac_sparsity,
        initials_for_algebra=initials_for_algebra,
        warnings=list(prepared_context.warnings),
        request=request,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )


def _bind_parameters_to_mechanism(mech: Any, names: List[str]) -> Dict[str, Any]:
    from kindred.core.equilibrium_rate_authority import require_step_entry_role_editable
    from kindred.core.rate_binding import RateBinding
    from kindred.core.simulator.step_indexing import lookup_step_param_target
    from kindred.core.validation import try_parse_callable_finite_float

    bindings: Dict[str, RateBinding] = {}

    for raw_name in names:
        name = str(raw_name or "").strip()
        target = lookup_step_param_target(mech, name)
        if target is None:
            continue
        kind, idx, role, entry = target
        if kind == "equilibrium":
            try:
                require_step_entry_role_editable(entry, role, parameter_name=name)
            except ValueError as exc:
                raise SimulationPreparationError("parameter_binding", str(exc)) from exc

        binding = bindings.get(name)
        if binding is None:
            if kind == "reaction" and role == "k":
                rxn = mech.reactions[idx]
                parsed, ok = try_parse_callable_finite_float(getattr(rxn, "rate", None))
                init = float(parsed) if ok else 1.0
            elif kind == "equilibrium" and role in {"kf", "kr"}:
                eq = mech.equilibria[idx]
                parsed, ok = try_parse_callable_finite_float(getattr(eq, role, None))
                init = float(parsed) if ok else 1.0
            elif kind == "equilibrium" and role == "Keq":
                eq = mech.equilibria[idx]
                meta = getattr(eq, "metadata", {}) or {}
                parsed, ok = try_parse_callable_finite_float(
                    meta.get(EquilibriumMetadataKeys.KEQ_INPUT),
                )
                if not ok:
                    raise SimulationPreparationError(
                        "parameter_binding",
                        (
                            f"Cannot bind {name!r}: equilibrium parameter has no explicit "
                            "equilibrium-constant source token."
                        ),
                    )
                init = float(parsed)
            else:
                init = 1.0
            binding = RateBinding(name=name, value=float(init))
            bindings[name] = binding

        if kind == "reaction" and role == "k":
            rxn = mech.reactions[idx]
            mech.reactions[idx] = replace(rxn, rate=binding)
        elif kind == "equilibrium" and role in {"kf", "kr"}:
            eq = mech.equilibria[idx]
            mech.equilibria[idx] = replace(eq, **{role: binding})
        elif kind == "equilibrium" and role == "Keq":
            eq = mech.equilibria[idx]
            meta = dict(getattr(eq, "metadata", {}) or {})
            meta[EquilibriumMetadataKeys.KEQ_INPUT] = binding
            mech.equilibria[idx] = replace(eq, metadata=meta)

    return bindings


def _implicit_equilibrium_constant_override_names(
    mechanism: Any,
    names: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> set[str]:
    from kindred.core.equilibrium_rate_authority import step_entry_role_editable
    from kindred.core.simulator.step_indexing import get_step_index_map

    requested = {str(name) for name in (names or ()) if str(name or "").strip()}
    excluded = {str(name) for name in (exclude or ()) if str(name or "").strip()}
    if not requested:
        return set()
    implicit_keq: set[str] = set()
    for entry in get_step_index_map(mechanism):
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
        if bool(step_entry_role_editable(entry, "Keq")):
            continue
        if keq_name in requested and keq_name not in excluded:
            implicit_keq.add(keq_name)
    return implicit_keq


def _reject_implicit_equilibrium_constant_overrides(
    mechanism: Any,
    names: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> None:
    requested_implicit_keq = _implicit_equilibrium_constant_override_names(
        mechanism,
        names,
        exclude=exclude,
    )
    if not requested_implicit_keq:
        return
    raise SimulationPreparationError(
        "parameter_algebra",
        (
            "Implicit equilibrium parameter(s) "
            + ", ".join(sorted(requested_implicit_keq))
            + " are not writable runtime or fitting parameters without an explicit equilibrium-constant source token; "
            "they are computed from current forward/reverse rates."
        ),
    )


def _internal_parameter_algebra_binding_names(
    mechanism: Any,
    requested_names: Iterable[str] = (),
) -> set[str]:
    from kindred.core.equilibrium_rate_authority import authority_fields_from_step_entry, step_entry_role_editable
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
        has_keq_param = bool(step_entry_role_editable(entry, "Keq"))
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


@dataclass
class _StateNetworkEnergyBindingState:
    units: Any
    temperature_K: float
    kappa: float
    degeneracy_ratio_fwd: float
    degeneracy_ratio_rev: float
    std_ratio_fwd: float
    std_ratio_rev: float
    dG_act_fwd_J_per_mol: float
    dG_eq_J_per_mol: float
    metadata: Dict[str, Any]

    def _sync_metadata(self) -> None:
        self.metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(self.dG_eq_J_per_mol)
        self.metadata["dG_act_fwd_J_per_mol"] = float(self.dG_act_fwd_J_per_mol)
        self.metadata["dG_act_rev_J_per_mol"] = float(self.dG_act_rev_J_per_mol())
        self.metadata["kf"] = float(self.kf())
        self.metadata["kr"] = float(self.kr())
        self.metadata["Keq"] = float(self.Keq())

    def dG_act_rev_J_per_mol(self) -> float:
        return float(self.dG_act_fwd_J_per_mol - self.dG_eq_J_per_mol)

    def _eyring_rate(self, dG_barrier_J_per_mol: float, *, degeneracy_ratio: float, std_ratio: float) -> float:
        from kindred.core.constants import R, h, k_B

        T = float(self.temperature_K)
        exponent = math.exp(-float(dG_barrier_J_per_mol) / (float(R) * T))
        prefactor = float(self.kappa) * float(k_B) * T / float(h)
        return float(prefactor * float(degeneracy_ratio) * exponent * float(std_ratio))

    def kf(self) -> float:
        return self._eyring_rate(
            self.dG_act_fwd_J_per_mol,
            degeneracy_ratio=self.degeneracy_ratio_fwd,
            std_ratio=self.std_ratio_fwd,
        )

    def kr(self) -> float:
        return self._eyring_rate(
            self.dG_act_rev_J_per_mol(),
            degeneracy_ratio=self.degeneracy_ratio_rev,
            std_ratio=self.std_ratio_rev,
        )

    def Keq(self) -> float:
        from kindred.core.kinetics import K_from_deltaG_eq

        return float(K_from_deltaG_eq(self.dG_eq_J_per_mol, self.temperature_K))

    def binding_value(self, role: str) -> float:
        if role == "dG_act_fwd":
            return float(self.units.from_jmol(self.dG_act_fwd_J_per_mol))
        if role == "dG_eq":
            return float(self.units.from_jmol(self.dG_eq_J_per_mol))
        if role == "kf":
            return float(self.kf())
        if role == "kr":
            return float(self.kr())
        if role == "Keq":
            return float(self.Keq())
        raise KeyError(f"Unknown state-network energy binding role: {role}")

    def apply_slider_value(self, role: str, value: float) -> None:
        if role == "dG_act_fwd":
            self.dG_act_fwd_J_per_mol = float(self.units.to_jmol(value))
        elif role == "dG_eq":
            self.dG_eq_J_per_mol = float(self.units.to_jmol(value))
        else:
            raise KeyError(f"Unknown state-network energy slider role: {role}")
        self._sync_metadata()


@dataclass
class _FastEquilibriumEnergyBindingState:
    units: Any
    temperature_K: float
    kf_fixed: float
    std_ratio: float
    dG_eq_J_per_mol: float
    metadata: Dict[str, Any]
    kf_getter: Optional[Callable[[], float]] = None
    kf_setter: Optional[Callable[[float], None]] = None

    def _sync_metadata(self) -> None:
        self.metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(self.dG_eq_J_per_mol)
        self.metadata["kf"] = float(self.kf())
        self.metadata["kr"] = float(self.kr())
        self.metadata["Keq"] = float(self.Keq())

    def Keq(self) -> float:
        from kindred.core.kinetics import K_from_deltaG_eq

        return float(K_from_deltaG_eq(self.dG_eq_J_per_mol, self.temperature_K))

    def kf(self) -> float:
        if self.kf_getter is not None:
            return float(self.kf_getter())
        return float(self.kf_fixed)

    def kr(self) -> float:
        return float(self.kf() / (self.Keq() * self.std_ratio))

    def binding_value(self, role: str) -> float:
        if role == "dG_eq_fast":
            return float(self.units.from_jmol(self.dG_eq_J_per_mol))
        if role == "kf":
            return float(self.kf())
        if role == "kr":
            return float(self.kr())
        if role == "Keq":
            return float(self.Keq())
        raise KeyError(f"Unknown fast-equilibrium energy binding role: {role}")

    def apply_slider_value(self, role: str, value: float) -> None:
        from kindred.core.constants import R

        value_f = float(value)
        if role == "dG_eq_fast":
            self.dG_eq_J_per_mol = float(self.units.to_jmol(value_f))
        elif role == "Keq":
            if not (math.isfinite(value_f) and value_f > 0.0):
                raise ValueError("Fast-equilibrium Keq must be positive and finite")
            self.dG_eq_J_per_mol = float(-float(R) * float(self.temperature_K) * math.log(value_f))
        elif role == "kr":
            if not (math.isfinite(value_f) and value_f > 0.0):
                raise ValueError("Fast-equilibrium kr must be positive and finite")
            Keq_value = float(self.kf() / (value_f * self.std_ratio))
            if not (math.isfinite(Keq_value) and Keq_value > 0.0):
                raise ValueError("Fast-equilibrium derived Keq must be positive and finite")
            self.dG_eq_J_per_mol = float(-float(R) * float(self.temperature_K) * math.log(Keq_value))
        elif role == "kf":
            if not (math.isfinite(value_f) and value_f > 0.0):
                raise ValueError("Fast-equilibrium kf must be positive and finite")
            if self.kf_setter is not None:
                self.kf_setter(value_f)
            self.kf_fixed = float(value_f)
        else:
            raise KeyError(f"Unknown fast-equilibrium energy slider role: {role}")
        self._sync_metadata()


class _StructuredEnergyBinding(RateBinding):
    def __init__(self, *, name: str, state: Any, role: str) -> None:
        super().__init__(name=str(name), value=float(state.binding_value(role)))
        self._state = state
        self._role = str(role)

    def __call__(self) -> float:
        current = float(self._state.binding_value(self._role))
        self.value = current
        return current

    def set(self, new_value: float) -> None:
        self._state.apply_slider_value(self._role, float(new_value))
        self.value = float(self._state.binding_value(self._role))


def _canonical_fast_equilibrium_side(stoich: Mapping[str, float]) -> str:
    def _fmt_coeff(value: float) -> str:
        rounded = round(float(value))
        if abs(float(value) - float(rounded)) < 1e-12:
            return str(int(rounded))
        return f"{float(value):g}"

    parts: List[str] = []
    for name in sorted(str(key) for key in stoich.keys()):
        coeff = float(stoich[name])
        if abs(coeff - 1.0) < 1e-12:
            parts.append(str(name))
        else:
            parts.append(f"{_fmt_coeff(coeff)}{name}")
    return "_".join(parts)


def _available_energy_binding_names(mechanism: Any) -> frozenset[str]:
    names: set[str] = set()
    for eq in getattr(mechanism, "equilibria", []) or []:
        metadata = dict(getattr(eq, "metadata", {}) or {})
        reactant = str(metadata.get("reactant") or "")
        product = str(metadata.get("product") or "")
        ts = str(metadata.get("ts") or "")
        if str(metadata.get("source") or "") == "state_network" and reactant and product and ts:
            names.add(f"dGact_fwd__{ts}__{reactant}__{product}")
            names.add(f"dG_eq__{ts}__{reactant}__{product}")
        fast_equilibrium = bool(metadata.get(EquilibriumMetadataKeys.FAST_EQUILIBRIUM, getattr(eq, "fast", False)))
        if fast_equilibrium:
            slug = _canonical_fast_equilibrium_side(getattr(eq, "stoich_forward", {}) or {})
            slug += "__" + _canonical_fast_equilibrium_side(getattr(eq, "stoich_back", {}) or {})
            names.add(f"dG_eq_fast__feq__{slug}")
    return frozenset(names)


def _install_energy_bindings(mechanism: Any, names: List[str]) -> Dict[str, Any]:
    from kindred.core.mechanism_metadata import MechanismMetadataView
    from kindred.core.rate_binding import RateBinding
    from kindred.core.units import UnitsModel
    from kindred.core.validation import try_parse_callable_finite_float
    from kindred.core.constants import R

    requested = {str(name) for name in (names or []) if str(name).strip()}
    if not requested:
        return {}

    mech_meta = MechanismMetadataView.from_metadata(getattr(mechanism, "metadata", {}) or {})
    units = UnitsModel(energy_unit=str(mech_meta.energy_unit), temperature_K=float(mech_meta.temperature_K))
    bindings: Dict[str, Any] = {}

    for index, eq in enumerate(getattr(mechanism, "equilibria", []) or []):
        metadata = dict(getattr(eq, "metadata", {}) or {})

        reactant = str(metadata.get("reactant") or "")
        product = str(metadata.get("product") or "")
        ts = str(metadata.get("ts") or "")
        if str(metadata.get("source") or "") == "state_network" and reactant and product and ts:
            act_name = f"dGact_fwd__{ts}__{reactant}__{product}"
            eq_name = f"dG_eq__{ts}__{reactant}__{product}"
            if act_name not in requested and eq_name not in requested:
                continue

            state = _StateNetworkEnergyBindingState(
                units=units,
                temperature_K=float(metadata.get("temperature_K") or mech_meta.temperature_K),
                kappa=float(metadata.get("kappa") or 1.0),
                degeneracy_ratio_fwd=float(metadata.get("degeneracy_ratio_fwd") or 1.0),
                degeneracy_ratio_rev=float(metadata.get("degeneracy_ratio_rev") or 1.0),
                std_ratio_fwd=float(metadata.get("std_conc_product_ts") or 1.0)
                / max(1e-300, float(metadata.get("std_conc_product_reactant") or 1.0)),
                std_ratio_rev=float(metadata.get("std_conc_product_ts") or 1.0)
                / max(1e-300, float(metadata.get("std_conc_product_product") or 1.0)),
                dG_act_fwd_J_per_mol=float(metadata.get("dG_act_fwd_J_per_mol") or 0.0),
                dG_eq_J_per_mol=float(metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) or 0.0),
                metadata=metadata,
            )
            state._sync_metadata()
            kf_binding = _StructuredEnergyBinding(name=f"{act_name}:kf", state=state, role="kf")
            kr_binding = _StructuredEnergyBinding(name=f"{eq_name}:kr", state=state, role="kr")
            Keq_binding = _StructuredEnergyBinding(name=f"{eq_name}:Keq", state=state, role="Keq")
            metadata[EquilibriumMetadataKeys.KEQ_INPUT] = Keq_binding
            mechanism.equilibria[index] = replace(
                eq,
                kf=kf_binding,
                kr=kr_binding,
                Keq=Keq_binding,
                metadata=metadata,
            )
            if act_name in requested:
                bindings[act_name] = _StructuredEnergyBinding(name=act_name, state=state, role="dG_act_fwd")
            if eq_name in requested:
                bindings[eq_name] = _StructuredEnergyBinding(name=eq_name, state=state, role="dG_eq")
            continue

        fast_equilibrium = bool(metadata.get(EquilibriumMetadataKeys.FAST_EQUILIBRIUM, getattr(eq, "fast", False)))
        if not fast_equilibrium:
            continue
        slug = _canonical_fast_equilibrium_side(getattr(eq, "stoich_forward", {}) or {})
        slug += "__" + _canonical_fast_equilibrium_side(getattr(eq, "stoich_back", {}) or {})
        slider_name = f"dG_eq_fast__feq__{slug}"
        if slider_name not in requested:
            continue

        kf_value = getattr(eq, "kf", None)
        if isinstance(kf_value, RateBinding):
            kf_fixed = float(kf_value())
            kf_getter = kf_value
            kf_setter = kf_value.set
        else:
            kf_fixed = float(kf_value if kf_value is not None else 1.0)
            kf_getter = None
            kf_setter = None
        dG_eq_raw = metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
        dG_eq_parsed, dG_eq_ok = try_parse_callable_finite_float(dG_eq_raw)
        if dG_eq_ok:
            dG_eq_J_per_mol = float(dG_eq_parsed)
        else:
            Keq_input_value, Keq_input_ok = try_parse_callable_finite_float(metadata.get(EquilibriumMetadataKeys.KEQ_INPUT))
            if Keq_input_ok and math.isfinite(Keq_input_value) and Keq_input_value > 0.0:
                dG_eq_J_per_mol = float(-float(R) * float(mech_meta.temperature_K) * math.log(float(Keq_input_value)))
            else:
                dG_eq_J_per_mol = 0.0
        state = _FastEquilibriumEnergyBindingState(
            units=units,
            temperature_K=float(mech_meta.temperature_K),
            kf_fixed=float(kf_fixed),
            kf_getter=kf_getter,
            kf_setter=kf_setter,
            std_ratio=max(1e-300, float(metadata.get("std_ratio") or 1.0)),
            dG_eq_J_per_mol=float(dG_eq_J_per_mol),
            metadata=metadata,
        )
        state._sync_metadata()
        kr_binding = _StructuredEnergyBinding(name=f"{slider_name}:kr", state=state, role="kr")
        Keq_binding = _StructuredEnergyBinding(name=f"{slider_name}:Keq", state=state, role="Keq")
        metadata[EquilibriumMetadataKeys.KEQ_INPUT] = Keq_binding
        mechanism.equilibria[index] = replace(
            eq,
            kr=kr_binding,
            Keq=Keq_binding,
            metadata=metadata,
        )
        bindings[slider_name] = _StructuredEnergyBinding(name=slider_name, state=state, role="dG_eq_fast")

    return bindings


def prepare_bound_mechanism(
    mechanism_text: str,
    param_names: List[str],
    *,
    temperature_K: float = 298.15,
    initials: Optional[Dict[str, float]] = None,
    use_advanced_dsl: bool = True,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
) -> BoundMechanism:
    """
    Parse the DSL once, bind selected parameters, and compile the RHS.

    Returns a BoundMechanism holding mutable RateBinding objects so callers can
    update parameter values without re-parsing or rebuilding the ODE system.
    """
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.units import UnitsModel
    from kindred.core.ode_builder import build_ode_rhs_from_mechanism

    initials = initials or {}
    units = UnitsModel(temperature_K=temperature_K)

    if not use_advanced_dsl:
        logger.debug("Legacy basic DSL mode requested; using advanced parser instead.")
    try:
        mechanism = parse_dsl_to_mechanism(mechanism_text, initials=initials, units=units)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parse", exc)
        ) from exc
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
            parse_parameter_algebra_spec_from_dsl_text,
        )
        from kindred.core.simulator.step_indexing import get_step_index_map

        mechanism_namespace = mechanism_parameter_namespace(mechanism)
        unresolved_intervention_schedule = parse_intervention_schedule_from_dsl(str(mechanism_text or ""))
        spec = parse_parameter_algebra_spec_from_dsl_text(
            mechanism_text,
            mechanism_namespace=mechanism_namespace,
        )
        namespace_info = mechanism_namespace.info_by_name
        declared_scalar_names = {
            str(stmt.name)
            for stmt in (spec.param_statements or [])
            if str(stmt.name) not in namespace_info
        }
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
        canonicalize_request_parameter_names(requested_parameter_partition, param_names or [])
        constrained = {
            stmt.name
            for stmt in (spec.param_statements or [])
            if str(stmt.name) in namespace_info
        }
        constrained_keq_targets = {
            str(name) for name in constrained if namespace_info[str(name)].role == "Keq"
        }
        constrained_mutable_targets = set(constrained) - constrained_keq_targets
        active_keq_names = set(constrained_keq_targets)
        requested = {
            _canonical_step_override_name(mechanism, str(name))
            for name in requested_parameter_partition.bindable_mechanism_parameter_names
            if str(name or "").strip()
        }
        from kindred.core.equilibrium_rate_authority import authority_fields_from_step_entry, step_entry_role_editable

        requested_implicit_keq: set[str] = set()
        k_derived = set()
        for entry in get_step_index_map(mechanism):
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
            has_keq_param = bool(step_entry_role_editable(entry, "Keq"))
            if not authority:
                raise SimulationPreparationError(
                    "parameter_algebra",
                    f"Equilibrium step {n} is missing normalized equilibrium_authority metadata.",
                )
            has_thermo_param = bool(authority.get("has_thermo_param") or has_keq_param)
            if has_keq_param:
                active_keq_names.add(keq_name)
            elif keq_name in requested and keq_name not in constrained_keq_targets:
                requested_implicit_keq.add(keq_name)
            if not has_thermo_param and keq_name not in active_keq_names:
                continue
            derive_rate = str(authority.get("derived_role") or "")
            if derive_rate not in {"kf", "kr"}:
                derive_rate = "kr"
            if derive_rate:
                k_derived.add(f"{derive_rate}{n}")

        if requested_implicit_keq:
            raise SimulationPreparationError(
                "parameter_algebra",
                (
                    "Implicit equilibrium parameter(s) "
                    + ", ".join(sorted(requested_implicit_keq))
                    + " are not writable fit parameters without an explicit equilibrium-constant source token; "
                    "they are computed from current forward/reverse rates."
                ),
            )
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
        requested_derived_rates = sorted(str(name) for name in (requested & k_derived))
        if requested_derived_rates:
            raise SimulationPreparationError(
                "parameter_algebra",
                (
                    "Derived equilibrium rate parameter(s) cannot be fitted directly "
                    "because Keq/algebra constraints overwrite them: "
                    + ", ".join(requested_derived_rates)
                ),
            )
        mech_bind_names = sorted(
            {
                str(n)
                for n in (
                    (requested - constrained_keq_targets - k_derived)
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
        from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
        from kindred.core.simulator.wegscheider_symbolic import UnresolvedWegscheiderCyclicityError

        _ = apply_parameter_algebra_to_mechanism(mechanism_text, mechanism=mechanism, require_mutable=True)
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
        for name, binding in scalar_bindings.items():
            bindings.setdefault(str(name), binding)
    bindings.update(energy_bindings)

    try:
        rhs = build_ode_rhs_from_mechanism(mechanism)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("ode_build", exc)
        ) from exc

    return BoundMechanism(
        mechanism=mechanism,
        rhs=rhs,
        bindings=bindings,
        species_names=species_names,
        y0=y0,
        param_names=sorted(set(bindings.keys())),
        mechanism_text=mechanism_text,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
    )


def prepare_fitting_objective_context(
    *,
    mechanism_text: str,
    param_names: List[str],
    t_exp: np.ndarray,
    target_species: str,
    temperature_K: float = 298.15,
    initials: Optional[Dict[str, float]] = None,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    prepare_func: Callable[..., object] | None = None,
) -> PreparedFittingObjectiveContext:
    from kindred.core.algebra.simulation_series import compile_algebra_observables
    from kindred.core.exceptions import ErrorContext
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    try:
        prepared_solver_config = _build_solver_config(
            solver_input=str(solver or DEFAULT_SOLVER_NAME),
            rtol=rtol,
            atol=atol,
            grid={"N": int(np.asarray(t_exp, dtype=float).reshape(-1).size)},
            use_sparse_jacobian=USE_SPARSE_JACOBIAN_DEFAULT,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        )
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("solver_config", exc)
        ) from exc
    if prepared_solver_config.solver_warning:
        logger.warning("Solver normalization: %s (requested=%r)", prepared_solver_config.solver_warning, str(solver))

    prepare = prepare_bound_mechanism if prepare_func is None else prepare_func
    try:
        bound = prepare(
            mechanism_text=mechanism_text,
            param_names=param_names,
            temperature_K=temperature_K,
            initials=initials,
            use_advanced_dsl=True,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        )
        requested_parameter_partition = partition_simulation_parameter_values(
            mechanism=bound.mechanism,
            parameter_overrides=None,
            unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
            requested_parameter_names=param_names or [],
            runtime_parameter_names=bound.bindings.keys(),
        )
        _raise_unowned_request_parameter_values(requested_parameter_partition)
        canonical_requested_param_names = canonicalize_request_parameter_names(
            requested_parameter_partition,
            param_names or [],
        )
        _reject_requested_algebra_owned_mechanism_parameters_for_fitting(
            bound.mechanism,
            requested_parameter_partition.bindable_mechanism_parameter_names,
        )
    except SimulationPreparationError as exc:
        raise _fit_simulation_error_from_preparation_error(exc) from exc
    except ValueError as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parameter_binding", exc)
        ) from exc
    if not isinstance(bound, BoundMechanism):
        raise TypeError("prepare_func must return BoundMechanism-compatible output.")

    try:
        prepared_run_context = _build_prepared_run_context(
            mechanism=bound.mechanism,
            solver_config=prepared_solver_config,
            allow_dynamic_binding_symbolic_snapshot=_mechanism_supports_dynamic_symbolic_snapshot(bound.mechanism),
        )
    except Exception as exc:
        if isinstance(exc, SimulationPreparationError):
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("temperature_schedule", exc)
        ) from exc
    try:
        intervention_schedule = _metadata_view_for_mechanism(bound.mechanism).intervention_schedule
        if intervention_schedule is not None:
            intervention_schedule.validate_species(bound.species_names)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("intervention_schedule", exc)
        ) from exc
    compiled_algebra = None
    algebra_observables: set[str] = set()
    algebra_text = prepared_run_context.algebra_text
    if algebra_text:
        try:
            compiled_algebra = compile_algebra_observables(
                str(algebra_text),
                mechanism_namespace=build_namespace_from_mechanism(bound.mechanism),
            )
        except Exception as exc:
            raise FitSimulationError(
                f"Failed to parse Algebra observables for fitting: {exc}",
                details={"fatal": True},
            ) from exc
        if compiled_algebra.time_ref_statements:
            stmt = compiled_algebra.time_ref_statements[0]
            raise FitSimulationError(
                "Algebra baseline references like [A](T0) are not supported for fitting (v1).",
                details={"fatal": True},
                context=ErrorContext(line=stmt.line, col=stmt.col, line_text=stmt.line_text),
            )
        algebra_observables = set(compiled_algebra.observable_names)

    target_is_species = target_species in getattr(bound.mechanism, "species", {})
    target_is_algebra = target_species in algebra_observables
    if not target_is_species and not target_is_algebra:
        raise FitSimulationError(
            f"Target species '{target_species}' not found in mechanism species or Algebra observables.",
            details={"fatal": True},
        )

    target_species_index = bound.species_names.index(target_species) if target_is_species else None
    request = SimulationRequest(
        rhs=bound.rhs,
        t_span=(float(t_exp[0]), float(t_exp[-1])),
        y0=bound.y0,
        solver=str(prepared_solver_config.solver),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        t_eval=np.asarray(t_exp, dtype=float).reshape(-1),
        **prepared_run_context.symbolic_jacobian.to_request_kwargs(),
        temperature_schedule=prepared_run_context.temperature_schedule,
        intervention_schedule=intervention_schedule,
        species_names=tuple(bound.species_names),
        symbolic_wegscheider_identity=(
            dict(getattr(bound.mechanism, "metadata", {}).get("symbolic_wegscheider_identity") or {})
            if isinstance(getattr(bound.mechanism, "metadata", {}), Mapping)
            and isinstance(getattr(bound.mechanism, "metadata", {}).get("symbolic_wegscheider_identity"), Mapping)
            else None
        ),
    )
    initials_for_algebra = {
        name: float(bound.y0[idx]) for idx, name in enumerate(bound.species_names)
    }
    return PreparedFittingObjectiveContext(
        bound=bound,
        requested_param_names=canonical_requested_param_names,
        request=request,
        target_species=str(target_species),
        target_is_species=bool(target_is_species),
        target_species_index=target_species_index,
        compiled_algebra=compiled_algebra,
        initials_for_algebra=initials_for_algebra,
        temperature_K=float(temperature_K),
        unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
        warnings=list(prepared_run_context.warnings),
    )


def build_prepared_simulation_func(
    *,
    mechanism_text: str,
    param_names: List[str],
    t_end: float,
    num_points: int,
    temperature_K: float = 298.15,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    use_sparse_jacobian: bool = USE_SPARSE_JACOBIAN_DEFAULT,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    initial_prefix: str = "init:",
) -> Callable[[Dict[str, float]], SimulationSeriesPayload]:
    from kindred.core.algebra.simulation_series import (
        CompiledAlgebraSeries,
        compile_algebra_observables,
        evaluate_compiled_algebra_series_for_simulation,
    )
    from kindred.core.algebra.errors import (
        AlgebraError,
        AlgebraNameError,
        AlgebraShadowError,
        AlgebraSyntaxError,
    )
    from kindred.core.exceptions import ErrorContext
    t_end = float(t_end)
    grid_n = max(2, int(num_points))
    initial_prefix = str(initial_prefix or "init:")

    prepared_solver_config = _build_solver_config(
        solver_input=str(solver or DEFAULT_SOLVER_NAME),
        rtol=rtol,
        atol=atol,
        grid={"N": int(grid_n)},
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    if prepared_solver_config.solver_warning:
        logger.warning(prepared_solver_config.solver_warning)

    initial_bound: BoundMechanism | None = None
    initial_parse_error: Exception | None = None
    try:
        initial_bound = prepare_bound_mechanism(
            mechanism_text=mechanism_text,
            param_names=list(param_names or []),
            temperature_K=float(temperature_K),
            initials={},
            use_advanced_dsl=True,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        )
        metadata_requested_partition = partition_simulation_parameter_values(
            mechanism=initial_bound.mechanism,
            parameter_overrides=None,
            unresolved_intervention_schedule=initial_bound.unresolved_intervention_schedule,
            requested_parameter_names=param_names or [],
            runtime_parameter_names=initial_bound.bindings.keys(),
        )
        _raise_unowned_request_parameter_values(
            metadata_requested_partition,
            allow_unbound_mechanism_parameters=True,
        )
        canonical_metadata_param_names = canonicalize_request_parameter_names(
            metadata_requested_partition,
            param_names or [],
        )
        intervention_schedule_fingerprint = normalized_intervention_schedule_fingerprint(
            initial_bound.unresolved_intervention_schedule,
            mechanism_namespace=build_namespace_from_mechanism(initial_bound.mechanism),
        )
    except SimulationPreparationError as exc:
        raise _fit_simulation_error_from_preparation_error(exc) from exc
    except Exception as exc:
        initial_parse_error = exc
        canonical_metadata_param_names = sorted({str(x) for x in (param_names or []) if str(x).strip()})
        intervention_schedule_fingerprint = ""

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256((mechanism_text or "").encode("utf-8")).hexdigest(),
        mechanism_text_len=len(mechanism_text or ""),
        param_names=sorted({str(x) for x in canonical_metadata_param_names if str(x).strip()}),
        t_end=float(t_end),
        num_points=int(grid_n),
        temperature_K=float(temperature_K),
        solver_requested=str(prepared_solver_config.solver_input),
        solver_normalized=str(prepared_solver_config.solver),
        solver_warning=(
            str(prepared_solver_config.solver_warning)
            if prepared_solver_config.solver_warning
            else None
        ),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        use_sparse_jacobian=bool(prepared_solver_config.use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        initial_prefix=str(initial_prefix),
        intervention_schedule_fingerprint=str(intervention_schedule_fingerprint),
    )

    bound: Optional[BoundMechanism] = None
    species_index: Dict[str, int] = {}
    temperature_schedule: Optional[TemperatureScheduleProtocol] = None
    intervention_schedule: Optional[InterventionSchedule] = None
    symbolic_jacobian = SymbolicJacobianExecution.absent()
    last_shared_fp: Optional[Tuple[Tuple[str, float], ...]] = None
    compiled_algebra: Optional[CompiledAlgebraSeries] = None

    def _ensure_prepared() -> None:
        nonlocal bound, species_index, temperature_schedule, intervention_schedule, symbolic_jacobian, compiled_algebra, prepared_meta
        if bound is not None:
            return
        if initial_bound is None:
            if initial_parse_error is not None:
                raise _fit_simulation_error_from_preparation_error(
                    _prepare_preparation_failure("parse", initial_parse_error)
                ) from initial_parse_error
            raise _fit_simulation_error_from_preparation_error(
                _prepare_preparation_failure("parse", "Prepared mechanism unavailable.")
            )
        bound = initial_bound
        try:
            requested_parameter_partition = partition_simulation_parameter_values(
                mechanism=bound.mechanism,
                parameter_overrides=None,
                unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
                requested_parameter_names=param_names or [],
                runtime_parameter_names=bound.bindings.keys(),
            )
            _raise_unowned_request_parameter_values(requested_parameter_partition)
            _reject_requested_algebra_owned_mechanism_parameters_for_fitting(
                bound.mechanism,
                requested_parameter_partition.bindable_mechanism_parameter_names,
            )
        except SimulationPreparationError as exc:
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        except ValueError as exc:
            raise _fit_simulation_error_from_preparation_error(
                _prepare_preparation_failure("parameter_binding", exc)
            ) from exc
        species_index = {name: idx for idx, name in enumerate(bound.species_names)}
        prepared_context = _build_prepared_run_context(
            mechanism=bound.mechanism,
            solver_config=prepared_solver_config,
            jacobian_func_override=symbolic_jacobian.jacobian_func,
            allow_dynamic_binding_symbolic_snapshot=_mechanism_supports_dynamic_symbolic_snapshot(bound.mechanism),
        )
        temperature_schedule = prepared_context.temperature_schedule
        symbolic_jacobian = prepared_context.symbolic_jacobian
        prepared_meta = _prepared_metadata_with_symbolic_jacobian(prepared_meta, symbolic_jacobian)
        simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]
        meta = getattr(bound.mechanism, "metadata", {}) or {}
        if isinstance(meta, Mapping) and isinstance(meta.get("symbolic_wegscheider_identity"), Mapping):
            prepared_meta = replace(
                prepared_meta,
                symbolic_wegscheider_identity=dict(meta.get("symbolic_wegscheider_identity") or {}),
            )
            simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]
        try:
            intervention_schedule = _metadata_view_for_mechanism(bound.mechanism).intervention_schedule
            if intervention_schedule is not None:
                intervention_schedule.validate_species(bound.species_names)
        except Exception as exc:
            raise FitSimulationError(
                f"Intervention schedule failed during fitting simulation preparation: {exc}",
                details={"fatal": True},
            ) from exc
        algebra_text = prepared_context.algebra_text
        if algebra_text:
            try:
                from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

                compiled_algebra = compile_algebra_observables(
                    str(algebra_text),
                    mechanism_namespace=build_namespace_from_mechanism(bound.mechanism),
                )
            except Exception as exc:
                raise FitSimulationError(
                    f"Failed to parse Algebra observables for fitting: {exc}",
                    details={"fatal": True},
                ) from exc
            if compiled_algebra.time_ref_statements:
                stmt = compiled_algebra.time_ref_statements[0]
                raise FitSimulationError(
                    "Algebra baseline references like [A](T0) are not supported for fitting (v1).",
                    details={"fatal": True},
                    context=ErrorContext(line=stmt.line, col=stmt.col, line_text=stmt.line_text),
                )

    def simulation_func(params: Dict[str, float]) -> SimulationSeriesPayload:
        nonlocal last_shared_fp, prepared_meta
        _ensure_prepared()
        assert bound is not None

        initial_overrides: Dict[str, float] = {}
        shared_values: Dict[str, float] = {}
        for key, raw_val in (params or {}).items():
            name = str(key)
            try:
                value = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise FitSimulationError(
                    f"Invalid parameter value for {name!r}: {raw_val!r}",
                    details={"fatal": True},
                ) from exc
            if name.startswith(initial_prefix):
                initial_overrides[name[len(initial_prefix) :]] = value
            else:
                shared_values[name] = value

        shared_fp = tuple(sorted((name, float(val)) for name, val in shared_values.items()))
        parameter_partition = partition_simulation_parameter_values(
            mechanism=bound.mechanism,
            parameter_overrides=shared_values,
            unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
            runtime_parameter_names=bound.bindings.keys(),
        )
        try:
            _raise_unowned_request_parameter_values(parameter_partition)
        except ValueError as exc:
            raise FitSimulationError(
                str(exc),
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc
        if shared_fp != last_shared_fp:
            for name, value in sorted(parameter_partition.mechanism_binding_values.items()):
                binding = bound.bindings.get(name)
                if binding is None:
                    continue
                try:
                    binding.set(float(value))
                except Exception as exc:
                    raise FitSimulationError(
                        f"Failed to update parameter binding {name!r}: {exc}",
                        details={"fatal": True},
                    ) from exc
            try:
                from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

                _ = apply_parameter_algebra_to_mechanism(
                    mechanism_text,
                    mechanism=bound.mechanism,
                    require_mutable=True,
                )
            except Exception as exc:
                raise FitSimulationError(f"Parameter algebra failed during global simulation: {exc}") from exc
            last_shared_fp = shared_fp

        current_intervention_schedule = intervention_schedule
        if bound.unresolved_intervention_schedule is not None:
            try:
                current_intervention_schedule = bound.unresolved_intervention_schedule.resolve_parameters(
                    parameter_partition.schedule_resolution_values
                )
                current_intervention_schedule.validate_species(bound.species_names)
            except InterventionScheduleError as exc:
                raise FitSimulationError(
                    f"Intervention schedule failed during fitting simulation: {exc}",
                    details={"fatal": True, "stage": "intervention_schedule"},
                ) from exc

        y0 = np.asarray(bound.y0, dtype=float).copy()
        for species_name, value in initial_overrides.items():
            idx = species_index.get(species_name)
            if idx is None:
                continue
            y0[idx] = float(value)

        current_symbolic_jacobian = symbolic_jacobian
        if (
            bool(prepared_solver_config.use_sparse_jacobian)
            and str(prepared_solver_config.solver).upper() in {"RADAU", "BDF"}
            and temperature_schedule is None
            and _mechanism_supports_dynamic_symbolic_snapshot(bound.mechanism)
        ):
            try:
                current_jacobian_func, symbolic_jacobian_identity = _bind_symbolic_jacobian_for_current_mechanism(
                    mechanism=bound.mechanism,
                    prepared_solver_config=prepared_solver_config,
                    temperature_K=float(temperature_K),
                )
                current_symbolic_jacobian = SymbolicJacobianExecution.supported(
                    jacobian_func=current_jacobian_func,
                    identity=symbolic_jacobian_identity,
                )
            except UnsupportedSymbolicExpressionError as exc:
                current_symbolic_jacobian = _symbolic_jacobian_for_bind_failure(bound.mechanism, exc)
            prepared_meta = _prepared_metadata_with_symbolic_jacobian(prepared_meta, current_symbolic_jacobian)
            simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]

        request = SimulationRequest(
            rhs=bound.rhs,
            t_span=(0.0, float(t_end)),
            y0=y0,
            solver=str(prepared_solver_config.solver),
            rtol=float(prepared_solver_config.rtol),
            atol=float(prepared_solver_config.atol),
            grid={"N": grid_n},
            **current_symbolic_jacobian.to_request_kwargs(),
            temperature_schedule=temperature_schedule,
            intervention_schedule=current_intervention_schedule,
            species_names=tuple(bound.species_names),
            symbolic_wegscheider_identity=(
                dict(getattr(bound.mechanism, "metadata", {}).get("symbolic_wegscheider_identity") or {})
                if isinstance(getattr(bound.mechanism, "metadata", {}), Mapping)
                and isinstance(getattr(bound.mechanism, "metadata", {}).get("symbolic_wegscheider_identity"), Mapping)
                else None
            ),
        )
        result = _solve_request(request)
        species_payload = {name: result.Y[idx, :].copy() for idx, name in enumerate(bound.species_names)}

        algebra_scalars: Dict[str, float] = {}
        if compiled_algebra is not None:
            try:
                initials_map = {name: float(y0[idx]) for idx, name in enumerate(bound.species_names)}
                species_series = {name: result.Y[idx, :] for idx, name in enumerate(bound.species_names)}
                algebra_series, algebra_scalars = evaluate_compiled_algebra_series_for_simulation(
                    bound.mechanism,
                    compiled_algebra,
                    t=result.t,
                    species_series=species_series,
                    initials=initials_map,
                    temperature_K=float(temperature_K),
                )
                for name, values in (algebra_series or {}).items():
                    if name in species_payload:
                        continue
                    species_payload[str(name)] = np.asarray(values, dtype=float).reshape(-1).copy()
            except AlgebraError as exc:
                is_fatal = isinstance(exc, (AlgebraNameError, AlgebraShadowError, AlgebraSyntaxError))
                raise FitSimulationError(
                    f"Algebra evaluation failed during fitting simulation: {exc}",
                    details={"fatal": bool(is_fatal)},
                    context=ErrorContext(line=exc.line, col=exc.col, line_text=exc.line_text),
                ) from exc
            except FitSimulationError:
                raise
            except Exception as exc:
                raise FitSimulationError(
                    f"Algebra evaluation failed during fitting simulation: {exc}",
                    details={"fatal": False},
                ) from exc

        return SimulationSeriesPayload(
            t=result.t.copy(),
            species=species_payload,
            algebra_scalars=dict(algebra_scalars),
        )

    simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]
    return simulation_func
