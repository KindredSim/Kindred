"""Simulation preparation for fitting and prepared/bound execution paths.

The parser-owned ``PreparedDslRequestSet`` is the admitted science owner. This
module coordinates typed parameter binding, solver/Jacobian preparation, and
RHS compilation for execution paths that consume it.
"""

from __future__ import annotations

import logging
from numbers import Real
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np

from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.exceptions import (
    FitSimulationError,
    SimulationCancelled,
)
from kindred.core.species import validate_concentration, validate_time_seconds
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.temperature import (
    TemperatureScheduleProtocol,
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
from kindred.core.simulator.solver_types import (
    SimulationRequest,
    SimulationCancellationControl,
)
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
)
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulation_preparation_stages import (
    PreparedSolverConfig as _PreparedSolverConfig,
    build_prepared_run_context,
)
from kindred.core.time_grid import build_time_grid

logger = logging.getLogger(__name__)


_PreparedParameterTarget = tuple[
    str,
    tuple[str, int, str, Mapping[str, object]],
]


def _build_parameter_target_map(
    mechanism: Any,
) -> dict[str, _PreparedParameterTarget]:
    """Build one admitted indexed/physical target map for a preparation."""

    namespace = build_namespace_from_mechanism(mechanism)
    step_roster = get_step_index_map(mechanism)
    targets: dict[str, _PreparedParameterTarget] = {}

    for item in namespace.ordered_items:
        source_index = item.source_index
        if source_index is None or not (0 <= int(source_index) < len(step_roster)):
            continue
        entry = step_roster[int(source_index)]
        kind = str(entry.get("kind") or "")
        index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
        try:
            index = int(entry[index_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Admitted parameter {item.canonical_name!r} has no valid target index."
            ) from exc
        role = str(item.info.role or "")
        if kind not in {"reaction", "equilibrium"} or not role:
            continue
        targets[str(item.canonical_name)] = (
            "indexed",
            (kind, index, role, entry),
        )

    for coordinate in namespace.source_coordinates:
        name = str(coordinate.canonical_name)
        if name in targets:
            continue
        source_index = coordinate.source_index
        if source_index is None or not (0 <= int(source_index) < len(step_roster)):
            continue
        entry = step_roster[int(source_index)]
        kind = str(entry.get("kind") or "")
        index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
        try:
            index = int(entry[index_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Admitted source coordinate {name!r} has no valid target index."
            ) from exc
        if kind not in {"reaction", "equilibrium"}:
            continue
        targets[name] = (
            "physical",
            (kind, index, str(coordinate.role), entry),
        )
    return targets


def _target_parts(
    target: _PreparedParameterTarget,
) -> tuple[str, str, int, str, Mapping[str, object]]:
    route, facts = target
    kind, index, role, entry = facts
    return str(route), str(kind), int(index), str(role), entry


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


__all__ = [
    "SimulationExecutionRequest",
    "PreparedSimulationRun",
    "SimulationPreparationError",
    "build_simulation_request_from_prepared_run",
    "build_simulation_execution_request",
    "prepare_simulation_worker_run",
    "fit_simulation_error_from_preparation_error",
    "prepare_preparation_failure",
    "raise_unowned_request_parameter_values",
    "reject_requested_algebra_owned_mechanism_parameters_for_fitting",
    "simulation_execution_requests_equal",
]


def _simulation_container_values_equal(left: object, right: object) -> bool:
    """Compare only the plain containers used by retained request values."""

    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
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

def simulation_execution_requests_equal(
    left: SimulationExecutionRequest,
    right: SimulationExecutionRequest,
) -> bool:
    """Compare one accepted owner identity and its typed target values."""

    return (
        left.prepared_dsl is right.prepared_dsl
        and left.species_names == right.species_names
        and left.y0 == right.y0
        and left.t_span == right.t_span
        and type(left.solver_config) is type(right.solver_config)
        and left.solver_config.solver_input == right.solver_config.solver_input
        and left.solver_config.solver == right.solver_config.solver
        and left.solver_config.rtol == right.solver_config.rtol
        and left.solver_config.atol == right.solver_config.atol
        and _simulation_container_values_equal(left.solver_config.grid, right.solver_config.grid)
        and left.solver_config.use_analytic_jacobian == right.solver_config.use_analytic_jacobian
        and left.requested_parameter_names == right.requested_parameter_names
        and left.candidate_dependent_intervention_parameter_names == right.candidate_dependent_intervention_parameter_names
        and _simulation_container_values_equal(left.parameter_overrides, right.parameter_overrides)
    )


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
    invalid_messages: dict[str, str],
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
    requested_parameter_names: Iterable[str] | None = None,
    declared_scalar_names: Iterable[str] | None = None,
    derived_scalar_names: Iterable[str] | None = None,
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
    derived_scalar_names = {
        str(name or "").strip()
        for name in derived_scalar_names or ()
        if str(name or "").strip()
    }
    for name in classification_names:
        claimed = False
        target = parameter_target_map.get(name)
        if target is not None:
            route, _kind, _index, _role, _entry = _target_parts(target)
            canonical_name = name
            if route == "indexed":
                try:
                    can_apply = _indexed_parameter_target_can_apply(
                        mechanism,
                        target,
                        canonical_name=name,
                    )
                except SimulationPreparationError:
                    # A derived indexed value can be a schedule readout.  It
                    # is not a writable request coordinate unless a value was
                    # actually supplied for it.
                    if name in schedule_names and name not in raw_values:
                        continue
                    raise
            else:
                can_apply = _role in {
                    "A",
                    "Ea",
                    "dG_act",
                    "dG_eq",
                    "kappa",
                }
            if can_apply:
                mechanism_names.add(canonical_name)
                mechanism_name_by_raw[name] = canonical_name
            else:
                if name in schedule_names and name not in raw_values:
                    continue
                unbound_mechanism_names.add(canonical_name)
                unbound_mechanism_name_by_raw[name] = canonical_name
            continue
        invalid_message = invalid_request_parameter_identifier_message(mechanism, name)
        if invalid_message is not None:
            invalid_messages[name] = invalid_message
            continue
        if name in raw_values and name in derived_scalar_names:
            invalid_messages[name] = (
                f"derived scalar parameter {name!r} cannot be used as direct runtime "
                "or schedule override authority."
            )
            continue
        if name in schedule_names:
            claimed = True
        if name in declared_scalar_names:
            scalar_names.add(name)
            claimed = True
        if not claimed:
            unknown_names.add(name)


def _indexed_parameter_target_can_apply(
    mechanism: Any,
    target: _PreparedParameterTarget,
    *,
    canonical_name: str,
) -> bool:
    """Return whether one indexed target has a prepared write authority."""

    _route, kind, index, role, entry = _target_parts(target)
    if kind == "equilibrium":
        _equilibrium_override_binding_target(
            mechanism,
            idx=index,
            role=role,
            entry=entry,
        )
        return True
    # Binding creates a target-local RateBinding when the parser supplied a
    # literal.  Current object mutability is therefore not an admission
    # criterion; only the admitted indexed role decides ownership.
    if kind == "reaction" and role == "k":
        source_coordinates = entry.get("source_coordinates") or {}
        if isinstance(source_coordinates, Mapping) and any(
            str(role) in {"A", "Ea", "dG_act"}
            for role in source_coordinates
        ):
            raise SimulationPreparationError(
                "parameter_overrides",
                f"mechanism parameter {canonical_name!r} is a derived rate readout and cannot be used as write authority.",
                parameter_names=(canonical_name,),
            )
        return True
    return False

def raise_unowned_request_parameter_values(
    *,
    raw_values: Mapping[str, float],
    mechanism_name_by_raw: Mapping[str, str],
    unbound_mechanism_name_by_raw: Mapping[str, str],
    unbound_mechanism_names: Iterable[str],
    unknown_names: Iterable[str],
    invalid_messages: Mapping[str, str],
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
    if unbound:
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
    request: SimulationExecutionRequest
    unresolved_intervention_schedule: InterventionSchedule | None = None
    cancellation_control: SimulationCancellationControl | None = None

    def __post_init__(self) -> None:
        y0 = np.asarray(self.y0, dtype=float).reshape(-1).copy()
        y0.setflags(write=False)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "species_names", tuple(self.species_names))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        if self.initials_for_algebra is not None:
            if not isinstance(self.initials_for_algebra, Mapping):
                raise TypeError("initials_for_algebra must be a mapping or None.")
            object.__setattr__(
                self,
                "initials_for_algebra",
                _freeze_prepared_value(self.initials_for_algebra),
            )
def build_simulation_request_from_prepared_run(
    prepared_run: PreparedSimulationRun,
    *,
    y0: np.ndarray,
    intervention_execution: InterventionExecutionContract,
    symbolic_jacobian: SymbolicJacobianExecution,
    cancellation_control: SimulationCancellationControl | None = None,
    t_eval: np.ndarray | None = None,
    candidate_evaluation: bool = False,
) -> SimulationRequest:
    request = prepared_run.request
    cancellation_control = (
        cancellation_control
        if cancellation_control is not None
        else prepared_run.cancellation_control
    )
    _raise_if_cancelled(cancellation_control)
    result = SimulationRequest(
        rhs=prepared_run.rhs,
        t_span=request.t_span,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        intervention_execution=intervention_execution,
        symbolic_jacobian=symbolic_jacobian,
        solver=request.solver_config.solver,
        rtol=request.solver_config.rtol,
        atol=request.solver_config.atol,
        grid=request.solver_config.grid,
        temperature_schedule=prepared_run.temperature_schedule,
        species_names=tuple(prepared_run.species_names),
        cancellation_control=cancellation_control,
        t_eval=t_eval,
        candidate_evaluation=candidate_evaluation,
    )
    _raise_if_cancelled(cancellation_control)
    return result


@dataclass(frozen=True, slots=True)
class SimulationExecutionRequest:
    """Exact direct request for one target of an accepted prepared owner."""

    prepared_dsl: PreparedDslRequestSet
    species_names: tuple[str, ...]
    y0: tuple[float, ...]
    t_span: tuple[float, float]
    solver_config: _PreparedSolverConfig
    requested_parameter_names: tuple[str, ...] = ()
    candidate_dependent_intervention_parameter_names: frozenset[str] = frozenset()
    parameter_overrides: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if type(self.prepared_dsl) is not PreparedDslRequestSet:
            raise TypeError("execution request prepared_dsl must be PreparedDslRequestSet.")
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
        schedule = self.prepared_dsl.parameterized_intervention_schedule_for_binding()
        contract = self.prepared_dsl.intervention_execution_contract()
        if schedule is None and not isinstance(contract, InterventionExecutionContract):
            raise TypeError("prepared DSL must retain one compiled intervention contract.")
        if schedule is not None and not schedule.is_parameterized:
            raise TypeError("prepared DSL declarative intervention state must be parameterized.")
        if schedule is not None and contract is not None:
            raise ValueError("prepared DSL intervention state cannot be both declarative and compiled.")
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
        object.__setattr__(self, "species_names", species_names)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "requested_parameter_names", requested_names)
        object.__setattr__(self, "candidate_dependent_intervention_parameter_names", candidate_dependent_names)
        object.__setattr__(self, "t_span", t_span)
        object.__setattr__(self, "parameter_overrides", overrides)


def _fresh_preparation_unresolved_intervention_schedule_for_request(
    execution_request: SimulationExecutionRequest,
) -> InterventionSchedule | None:
    return execution_request.prepared_dsl.parameterized_intervention_schedule_for_binding()


def _expanded_intervention_schedule_dependency_metadata(
    schedule: InterventionSchedule | None,
    *,
    parameter_algebra_spec: Any,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> InterventionScheduleDependencyMetadata:
    """Expand direct schedule names through the admitted Algebra dependency map."""

    direct_metadata = intervention_schedule_parameter_dependencies(schedule)
    if not direct_metadata:
        return direct_metadata
    admitted_dependencies = getattr(parameter_algebra_spec, "parameter_dependencies", {})
    if not isinstance(admitted_dependencies, Mapping):
        admitted_dependencies = {}

    def canonical_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        target = parameter_target_map.get(name)
        return name if target is not None else name

    expanded: dict[tuple[int, int | None, str], tuple[str, ...]] = {}
    for identity, raw_names in direct_metadata.items():
        leaves: set[str] = set()
        visiting: set[str] = set()

        def visit(raw_name: object) -> None:
            name = canonical_name(raw_name)
            if name in visiting:
                # Admission has already rejected cycles; this keeps the
                # consumer fail-closed if supplied typed state is corrupted.
                raise ValueError(
                    "Parameter algebra dependency map contains a cycle."
                )
            dependencies = admitted_dependencies.get(name)
            if dependencies is None:
                leaves.add(name)
                return
            dependencies_tuple = tuple(str(item).strip() for item in dependencies)
            if not dependencies_tuple:
                leaves.add(name)
                return
            visiting.add(name)
            for dependency in dependencies_tuple:
                if dependency:
                    visit(dependency)
            visiting.remove(name)

        for raw_name in raw_names:
            visit(raw_name)
        expanded[identity] = tuple(sorted(leaves))
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
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
    parameter_algebra_spec: Any | None = None,
) -> frozenset[str]:
    if not unresolved_intervention_schedule.is_parameterized:
        return frozenset()
    retained = execution_request.candidate_dependent_intervention_parameter_names
    if retained:
        # A candidate request carries the fitted values explicitly.  Resolve
        # it during this preparation while the admitted target map is alive;
        # only the base Fit request (which has no candidate values yet) keeps
        # the declarative schedule deferred.
        supplied = {
            str(name).strip()
            for name in dict(execution_request.parameter_overrides or {})
            if str(name).strip()
        }
        if retained.issubset(supplied):
            return frozenset()
        return retained
    requested = execution_request.requested_parameter_names
    supplied = set(dict(execution_request.parameter_overrides or {}).keys())
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    if not requested or not referenced:
        return frozenset()

    if parameter_algebra_spec is None:
        parameter_algebra_spec = (
            execution_request.prepared_dsl.parameter_algebra_spec_for_execution()
        )

    def canonical_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        target = parameter_target_map.get(name)
        return name if target is not None else name

    requested_names = {
        canonical_name(name)
        for name in requested
        if str(name).strip()
    }
    supplied_names = {canonical_name(name) for name in supplied if str(name).strip()}
    expanded_metadata = _expanded_intervention_schedule_dependency_metadata(
        unresolved_intervention_schedule,
        parameter_algebra_spec=parameter_algebra_spec,
        parameter_target_map=parameter_target_map,
    )
    deferred: set[str] = set()
    for identity, direct_names in intervention_schedule_parameter_dependencies(
        unresolved_intervention_schedule
    ).items():
        expanded_names = set(expanded_metadata.get(identity, ()))
        if not expanded_names.intersection(requested_names - supplied_names):
            continue
        deferred.update(str(raw_name) for raw_name in direct_names)
    return frozenset(deferred)


def build_simulation_execution_request(
    *,
    prepared_dsl: PreparedDslRequestSet,
    mechanism_source: Any,
    initial_values: Iterable[tuple[str, float]],
    t_span: tuple[float, float],
    solver_config: _PreparedSolverConfig,
    parameter_values: Mapping[str, float] | None = None,
) -> SimulationExecutionRequest:
    """Capture, isolate, and compile one direct Run/Preview target."""

    from kindred.core.mechanism_source import (
        MechanismAuthoringSource,
        source_matches_analysis,
    )

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    if not isinstance(mechanism_source, MechanismAuthoringSource):
        raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
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
            t_span[0],
            field="execution t_start",
        )
        t_end = validate_time_seconds(
            t_span[1],
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
    species_names = tuple(str(name) for name in prepared_dsl.analysis.species_names)
    if not species_names:
        raise ValueError("prepared execution requires at least one species.")
    if set(normalized_initials) != set(species_names):
        raise ValueError(
            "direct execution initial values must cover every accepted species exactly once."
        )
    request = SimulationExecutionRequest(
        prepared_dsl=prepared_dsl,
        species_names=species_names,
        y0=tuple(normalized_initials[name] for name in species_names),
        t_span=(t_start, t_end),
        solver_config=solver_config,
        requested_parameter_names=(),
        parameter_overrides=normalized_parameters,
    )
    return request


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


def _parameter_algebra_scalar_authority_names(spec: Any) -> tuple[frozenset[str], frozenset[str]]:
    dependencies = getattr(spec, "parameter_dependencies", None)
    assignment_kind = getattr(spec, "parameter_assignment_kind", None)
    if not isinstance(dependencies, Mapping) or not callable(assignment_kind):
        return frozenset(), frozenset()
    independent: set[str] = set()
    derived: set[str] = set()
    for raw_name in dependencies:
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            kind = str(assignment_kind(name))
        except KeyError:
            continue
        if kind == "independent scalar":
            independent.add(name)
        elif kind == "derived scalar":
            derived.add(name)
    return frozenset(independent), frozenset(derived)


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
    parameter_target_map: Mapping[str, _PreparedParameterTarget] | None = None,
    mechanism: Any | None = None,
) -> Dict[str, float] | None:
    values: Dict[str, float] = {
        str(name): float(value)
        for name, value in (raw_values or {}).items()
        if str(name).strip()
    }
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    if mechanism is not None:
        if referenced:
            if parameter_target_map is None:
                raise SimulationPreparationError(
                    "intervention_schedule",
                    "Intervention schedule resolution requires the admitted preparation target map.",
                )
            values.update(
                _effective_mechanism_schedule_values(
                    mechanism,
                    names={str(name) for name in referenced if str(name)},
                    parameter_target_map=parameter_target_map,
                )
            )
            values.update(_scalar_schedule_resolution_values_from_mechanism(mechanism))
    return values if values else None


def _effective_mechanism_schedule_values(
    mechanism: Any,
    *,
    names: Iterable[str],
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> dict[str, float]:
    """Read indexed schedule coordinates from the effective rate authority."""

    from kindred.core.equilibrium_rate_authority import (
        effective_equilibrium_keq,
        effective_equilibrium_reverse_rate,
    )
    from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator
    from kindred.core.mechanism_metadata import MechanismMetadataView
    from kindred.core.kinetics import (
        ArrheniusRateModel,
        EyringRateModel,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
    )
    from kindred.core.units import UnitsModel

    target_map = parameter_target_map
    requested = {str(name).strip() for name in names if str(name).strip()}
    if not requested:
        return {}
    temperature = float(
        MechanismMetadataView.from_metadata(
            getattr(mechanism, "metadata", {}) or {}
        ).temperature_K
    )
    coefficients = build_effective_rate_coefficient_evaluator(mechanism)(temperature)
    reaction_count = len(getattr(mechanism, "reactions", ()) or ())
    reactions = getattr(mechanism, "reactions", ()) or ()
    equilibria = getattr(mechanism, "equilibria", ()) or ()
    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    units = UnitsModel(energy_unit=metadata.energy_unit)
    values: dict[str, float] = {}
    for raw_name in requested:
        target = target_map.get(raw_name)
        if target is None:
            continue
        route, kind, index, role, _entry = _target_parts(target)
        if route == "physical":
            if role == "dG_eq" and kind == "equilibrium":
                raw_dg = dict(
                    getattr(equilibria[index], "metadata", {}) or {}
                ).get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
                try:
                    value = units.from_jmol(float(raw_dg))
                except (TypeError, ValueError, OverflowError):
                    continue
                if np.isfinite(value):
                    values[raw_name] = float(value)
                continue
            if kind == "reaction":
                if not 0 <= index < len(reactions):
                    continue
                model = forward_rate_model_for_reaction(
                    reactions[index],
                    default_standard_conc_M=float(metadata.standard_conc_M),
                )
            elif kind == "equilibrium":
                if not 0 <= index < len(equilibria):
                    continue
                model = forward_rate_model_for_equilibrium(
                    equilibria[index],
                    temperature_K=None,
                    default_standard_conc_M=float(metadata.standard_conc_M),
                )
            else:
                continue
            value: float | None = None
            if role == "A" and isinstance(model, ArrheniusRateModel):
                value = float(model.A)
            elif role == "Ea" and isinstance(model, ArrheniusRateModel):
                value = float(units.from_jmol(model.Ea_J_per_mol))
            elif role == "dG_act" and isinstance(model, EyringRateModel):
                value = float(units.from_jmol(model.dG_act_J_per_mol))
            elif role == "kappa" and isinstance(model, EyringRateModel):
                value = float(model.kappa)
            if value is not None and np.isfinite(value):
                values[raw_name] = value
            continue
        if route != "indexed":
            continue
        if kind == "reaction" and role == "k":
            values[raw_name] = float(coefficients.forward[index])
        elif kind == "equilibrium":
            coefficient_index = reaction_count + index
            if role == "kf":
                values[raw_name] = float(coefficients.forward[coefficient_index])
            elif role == "kr":
                reverse = effective_equilibrium_reverse_rate(
                    equilibria[index],
                    temperature_K=temperature,
                )
                if reverse is not None:
                    values[raw_name] = float(reverse)
            elif role == "Keq":
                equilibrium_constant = effective_equilibrium_keq(
                    equilibria[index],
                    temperature_K=temperature,
                )
                if equilibrium_constant is not None:
                    values[raw_name] = float(equilibrium_constant)
    return values


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
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
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
        and parameter_target_map.get(str(canonical_name), None) is not None
        and _target_parts(parameter_target_map[str(canonical_name)])[0] == "indexed"
    }
    scalar_override_values = {
        str(name): float(raw_values[name])
        for name in scalar_parameter_names
        if name in raw_values
    }
    # Admit and construct every indexed replacement before publishing any
    # physical/scalar/indexed value.  This guarantees a later invalid target
    # cannot leave an earlier target mutated in the isolated shell.
    from kindred.core.simulator.parameter_algebra import (
        _admit_mechanism_parameter_value,
    )
    from kindred.core.kinetics import (
        ConstantRateModel,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
    )
    mechanism_metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    reaction_replacements: dict[int, object] = {}
    equilibrium_replacements: dict[int, object] = {}
    indexed_values: list[tuple[str, str, int, str, float, Mapping[str, object]]] = []
    for name, value in sorted(
        (str(name), float(value)) for name, value in mechanism_binding_values.items()
    ):
        target = parameter_target_map.get(name)
        if target is None:
            continue
        route, kind, index, role, entry = _target_parts(target)
        if route != "indexed":
            continue
        target_role = str(role)
        if kind == "equilibrium":
            target_role, _ = _equilibrium_override_binding_target(
                mechanism,
                idx=index,
                role=target_role,
                entry=entry,
            )
        admitted = _admit_mechanism_parameter_value(
            mechanism,
            name,
            value,
            parameter_target_map=parameter_target_map,
        )
        indexed_values.append((name, kind, index, target_role, admitted, entry))

    def _new_indexed_binding(
        *,
        name: str,
        model: object,
        value: float,
    ) -> RateBinding:
        if isinstance(model, ConstantRateModel):
            return RateBinding(name=name, value=float(value))
        binding = ForwardRateBinding(
            name=name,
            model=model,
            reference_temperature_K=float(mechanism_metadata.temperature_K),
        )
        binding.set(float(value))
        return binding

    for name, kind, index, role, value, entry in indexed_values:
        if kind == "reaction" and role == "k":
            reaction = reaction_replacements.get(index, mechanism.reactions[index])
            model = forward_rate_model_for_reaction(
                reaction,
                default_standard_conc_M=float(mechanism_metadata.standard_conc_M),
            )
            reaction_replacements[index] = replace(
                reaction,
                rate=_new_indexed_binding(
                    name=name,
                    model=model,
                    value=value,
                ),
            )
            continue
        if kind != "equilibrium" or role not in {"kf", "kr", "Keq"}:
            raise SimulationPreparationError(
                "parameter_overrides",
                f"Unsupported mechanism parameter {name!r}.",
                parameter_names=(name,),
            )
        equilibrium = equilibrium_replacements.get(index, mechanism.equilibria[index])
        if role in {"kf", "kr"}:
            if role == "kf":
                model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=float(mechanism_metadata.temperature_K),
                    default_standard_conc_M=float(mechanism_metadata.standard_conc_M),
                )
                equilibrium_replacements[index] = replace(
                    equilibrium,
                    kf=_new_indexed_binding(
                        name=name,
                        model=model,
                        value=value,
                    ),
                )
            else:
                equilibrium_replacements[index] = replace(
                    equilibrium,
                    kr=RateBinding(name=name, value=float(value)),
                )
            continue
        equilibrium_metadata = dict(getattr(equilibrium, "metadata", {}) or {})
        equilibrium_metadata[EquilibriumMetadataKeys.KEQ_INPUT] = RateBinding(
            name=name,
            value=float(value),
        )
        equilibrium_replacements[index] = replace(
            equilibrium,
            metadata=equilibrium_metadata,
        )

    physical_values = {
        str(canonical_name): float(raw_values[raw_name])
        for raw_name, canonical_name in mechanism_parameter_name_by_raw.items()
        if raw_name in raw_values
        and parameter_target_map.get(str(canonical_name), None) is not None
        and _target_parts(parameter_target_map[str(canonical_name)])[0] == "physical"
        and _target_parts(parameter_target_map[str(canonical_name)])[3]
        in {"A", "Ea", "dG_act", "dG_eq", "kappa"}
    }
    if physical_values:
        try:
            from kindred.core.simulator.parameter_algebra import (
                apply_source_coordinate_values_to_mechanism,
            )

            apply_source_coordinate_values_to_mechanism(
                mechanism,
                physical_values,
                parameter_target_map=parameter_target_map,
            )
        except Exception as exc:
            raise SimulationPreparationError(
                "parameter_overrides",
                str(exc),
                parameter_names=tuple(sorted(physical_values)),
            ) from exc
    for idx, replacement in reaction_replacements.items():
        mechanism.reactions[idx] = replacement
    for idx, replacement in equilibrium_replacements.items():
        mechanism.equilibria[idx] = replacement

    override_applied = bool(
        physical_values or indexed_values or scalar_override_values
    )
    if override_applied:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )

        apply_parameter_algebra_spec_to_mechanism(
            parameter_algebra_spec,
            mechanism=mechanism,
            require_mutable=True,
            parameter_target_map=parameter_target_map,
            scalar_parameter_overrides=scalar_override_values,
        )
    return None


def _canonical_step_override_name(
    mechanism: Any,
    name: str,
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget] | None = None,
) -> str:
    name_s = str(name or "").strip()
    if not name_s:
        return ""
    if parameter_target_map is not None:
        target = parameter_target_map.get(name_s)
        if target is not None:
            return name_s
        return name_s
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
    from kindred.core.equilibrium_rate_authority import (
        normalize_existing_equilibrium_rate_authority,
        step_entry_authored_role_is_editable,
    )

    eq = mechanism.equilibria[int(idx)]
    editable = step_entry_authored_role_is_editable(entry, str(role))
    if editable is False:
        name = _equilibrium_step_parameter_name(entry, str(role))
        raise SimulationPreparationError(
            "parameter_overrides",
            f"equilibrium parameter {name} is a derived readout and cannot be used as write authority.",
            parameter_names=(name,),
        )
    if editable is True:
        return str(role), _equilibrium_step_parameter_name(entry, str(role))
    authority = normalize_existing_equilibrium_rate_authority(eq)
    if not authority.authored_role_is_editable(str(role)):
        name = _equilibrium_step_parameter_name(entry, str(role))
        raise SimulationPreparationError(
            "parameter_overrides",
            f"equilibrium parameter {name} is a derived readout and cannot be used as write authority.",
            parameter_names=(name,),
        )
    return str(role), _equilibrium_step_parameter_name(entry, str(role))


def _apply_execution_parameter_algebra(
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
    require_mutable: bool,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> None:
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
    )
    apply_parameter_algebra_spec_to_mechanism(
        parameter_algebra_spec,
        mechanism=mechanism,
        require_mutable=bool(require_mutable),
        parameter_target_map=parameter_target_map,
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
    - Retaining the admitted DSL temperature schedule through execution
    - Parameter algebra application
    - Generated symbolic Jacobian construction when supported by the solver path
    """

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
    temperature_schedule_override = execution_request.prepared_dsl.temperature_schedule_for_execution()
    parameter_algebra_spec = execution_request.prepared_dsl.parameter_algebra_spec_for_execution()
    require_mutable = True
    parameter_target_map = _build_parameter_target_map(mechanism)

    _apply_execution_parameter_algebra(
        mechanism=mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
        require_mutable=bool(require_mutable),
        parameter_target_map=parameter_target_map,
    )
    _raise_if_cancelled(cancellation_control)

    unresolved_intervention_schedule = (
        _fresh_preparation_unresolved_intervention_schedule_for_request(
            execution_request
        )
    )
    request_parameter_values: Mapping[str, float] | None = None
    if (
        bool(execution_request.parameter_overrides)
        or bool(
            unresolved_intervention_schedule is not None
            and unresolved_intervention_schedule.is_parameterized
        )
    ):
        try:
            request_parameter_values = _bind_request_parameter_overrides(
                mechanism=mechanism,
                parameter_algebra_spec=parameter_algebra_spec,
                parameter_overrides=execution_request.parameter_overrides,
                unresolved_intervention_schedule=unresolved_intervention_schedule,
                parameter_target_map=parameter_target_map,
            )
        except SimulationPreparationError:
            raise
    _raise_if_cancelled(cancellation_control)

    intervention_execution = execution_request.prepared_dsl.intervention_execution_contract()
    if unresolved_intervention_schedule is not None:
        if _schedule_parameters_deferred_to_prepared_request(
            execution_request,
            unresolved_intervention_schedule,
            parameter_target_map=parameter_target_map,
            parameter_algebra_spec=parameter_algebra_spec,
        ):
            unresolved_intervention_schedule.validate_species(species_names)
        else:
            try:
                dependency_metadata = _expanded_intervention_schedule_dependency_metadata(
                    unresolved_intervention_schedule,
                    parameter_algebra_spec=parameter_algebra_spec,
                    parameter_target_map=parameter_target_map,
                )
                resolved_schedule = _resolve_intervention_schedule_for_request(
                    unresolved_intervention_schedule,
                    parameter_values=_schedule_resolution_values_for_request(
                        unresolved_intervention_schedule,
                        raw_values=request_parameter_values,
                        parameter_target_map=parameter_target_map,
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
    request = execution_request

    return PreparedSimulationRun(
        mechanism=mechanism,
        rhs=rhs,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        species_names=list(species_names),
        temperature_schedule=temperature_schedule,
        parameter_algebra_spec=parameter_algebra_spec,
        intervention_execution=intervention_execution,
        symbolic_jacobian=runtime_symbolic_jacobian,
        initials_for_algebra=initials_for_algebra,
        warnings=list(preparation_warnings),
        request=request,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        cancellation_control=cancellation_control,
    )


def _bind_parameters_to_mechanism(
    mech: Any,
    names: Iterable[str],
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> None:
    from kindred.core.kinetics import (
        ConstantRateModel,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
    )
    from kindred.core.validation import try_parse_callable_finite_float

    bindings: Dict[str, RateBinding] = {}
    reaction_replacements: dict[int, object] = {}
    equilibrium_replacements: dict[int, object] = {}
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
        target_info = parameter_target_map.get(name)
        if target_info is None:
            continue
        route, kind, idx, role, entry = _target_parts(target_info)
        if route != "indexed":
            continue
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
                        # Seed only the newly constructed binding.  Do not
                        # mutate the mechanism's current binding until every
                        # requested replacement has been constructed.
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
            rxn = reaction_replacements.get(idx, mech.reactions[idx])
            reaction_replacements[idx] = replace(rxn, rate=binding)
        elif kind == "equilibrium" and target_role in {"kf", "kr"}:
            eq = equilibrium_replacements.get(idx, mech.equilibria[idx])
            equilibrium_replacements[idx] = replace(eq, **{target_role: binding})
        elif kind == "equilibrium" and target_role == "Keq":
            eq = equilibrium_replacements.get(idx, mech.equilibria[idx])
            meta = dict(getattr(eq, "metadata", {}) or {})
            meta[EquilibriumMetadataKeys.KEQ_INPUT] = binding
            equilibrium_replacements[idx] = replace(eq, metadata=meta)

    # Publish only after all requested targets have a concrete replacement.
    # This keeps a later binding/admission failure from partially mutating the
    # isolated mechanism shell.
    for idx, replacement in reaction_replacements.items():
        mech.reactions[idx] = replacement
    for idx, replacement in equilibrium_replacements.items():
        mech.equilibria[idx] = replacement


def _bind_request_parameter_overrides(
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
    parameter_overrides: Mapping[str, float] | None,
    unresolved_intervention_schedule: Any,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> Mapping[str, float]:
    """Bind one request through the fixed preparation-owned parameter path."""

    declared_scalar_names, derived_scalar_names = (
        _parameter_algebra_scalar_authority_names(
            parameter_algebra_spec,
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
        invalid_messages=invalid_messages,
        parameter_target_map=parameter_target_map,
        declared_scalar_names=declared_scalar_names,
        derived_scalar_names=derived_scalar_names,
    )
    mechanism_override_names = sorted(
        set(mechanism_names) | set(unbound_mechanism_names)
    )
    internal_names = _internal_parameter_algebra_binding_names(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
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
        parameter_target_map=parameter_target_map,
    )
    # Bindings are published only after all request values have been admitted
    # and every requested replacement has been constructed.  In particular,
    # a later invalid indexed value cannot leave an earlier binding-only
    # replacement visible on the isolated mechanism shell.
    _bind_parameters_to_mechanism(
        mechanism,
        sorted(set(mechanism_override_names) | set(internal_names)),
        parameter_target_map=parameter_target_map,
    )
    return dict(raw_values)


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
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> None:
    requested = {
        _canonical_step_override_name(
            mechanism,
            str(name),
            parameter_target_map=parameter_target_map,
        )
        for name in (requested_names or ())
        if str(name or "").strip()
    }
    target_info_by_name = {
        str(name): target for name, target in parameter_target_map.items()
    }
    algebra_owned = sorted(
        str(name)
        for name in (
            requested
            & _internal_parameter_algebra_binding_names(
                mechanism,
                parameter_algebra_spec=parameter_algebra_spec,
            )
        )
        if (
            str(name) in target_info_by_name
            and _target_parts(target_info_by_name[str(name)])[3]
            in {"k", "kf", "kr"}
        )
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
