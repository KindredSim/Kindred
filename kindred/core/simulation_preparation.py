"""Simulation preparation for fitting and prepared/bound execution paths.

The parser-owned ``PreparedDslRequestSet`` is the admitted science owner. This
module coordinates typed parameter binding, solver/Jacobian preparation, and
RHS compilation for execution paths that consume it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np

from kindred.core.exceptions import (
    FitSimulationError,
    SimulationCancelled,
    located,
)
from kindred.core.units import OrdinaryParameterDomainError
from kindred.core.temperature import TemperatureSchedule
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
    freeze_retained_value,
)
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.simulator.parameter_coordinates import (
    ParameterTarget,
    build_parameter_target_map,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulation_preparation_stages import (
    PreparedSolverConfig as _PreparedSolverConfig,
    build_prepared_run_context,
)
from kindred.core.time_grid import build_time_grid

logger = logging.getLogger(__name__)


__all__ = [
    "SimulationExecutionRequest",
    "PreparedSimulationRun",
    "SimulationPreparationError",
    "build_simulation_request_from_prepared_run",
    "build_simulation_execution_request",
    "prepare_simulation_run",
    "fit_simulation_error_from_preparation_error",
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
        and left.use_authored_stops == right.use_authored_stops
        and left.requested_parameter_names == right.requested_parameter_names
        and _simulation_container_values_equal(left.parameter_overrides, right.parameter_overrides)
    )


class SimulationPreparationError(RuntimeError):
    """
    Error raised when preparing a scientific execution request fails.

    Its text is the user-facing message; ``stage`` and ``details`` are
    structured facts for reports.
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
                str(key): freeze_retained_value(value)
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
    temperature_schedule: TemperatureSchedule | None
    parameter_algebra_spec: Any
    intervention_execution: InterventionExecutionContract | None
    symbolic_jacobian: SymbolicJacobianExecution
    initials_for_algebra: Mapping[str, float]
    request: SimulationExecutionRequest
    unresolved_intervention_schedule: InterventionSchedule | None = None
    cancellation_control: SimulationCancellationControl | None = None
    stop_observable_value: Callable[[str, float, np.ndarray, float], float] | None = None
    parameter_values_at_temperature: Callable[[float, tuple[str, ...]], Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        y0 = np.asarray(self.y0, dtype=float).reshape(-1).copy()
        y0.setflags(write=False)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "species_names", tuple(self.species_names))
        object.__setattr__(
            self,
            "initials_for_algebra",
            freeze_retained_value(self.initials_for_algebra),
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
        reference_temperature_K=float(request.prepared_dsl.analysis.temperature_K),
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
        stop_observable_value=prepared_run.stop_observable_value,
        parameter_values_at_temperature=prepared_run.parameter_values_at_temperature,
        intervention_lines=request.prepared_dsl.analysis.intervention_lines,
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
    parameter_overrides: Mapping[str, float] | None = None
    use_authored_stops: bool = False

    def __post_init__(self) -> None:
        # Its producers pass admitted values: initial concentrations from the
        # document or Slider and Scan admission (exact zeros for Fit), the span
        # from Run settings or Scan and Fit admission, and override values from
        # Slider, Scan or Fit candidate admission.  The request keeps its own
        # copy of the caller's override mapping.
        if self.parameter_overrides is not None:
            object.__setattr__(self, "parameter_overrides", dict(self.parameter_overrides))


def _fresh_preparation_unresolved_intervention_schedule_for_request(
    execution_request: SimulationExecutionRequest,
) -> InterventionSchedule | None:
    schedule = execution_request.prepared_dsl.parameterized_intervention_schedule_for_binding()
    return schedule if schedule is None or execution_request.use_authored_stops else schedule.without_stops()


def _expanded_intervention_schedule_dependency_metadata(
    schedule: InterventionSchedule | None,
    *,
    parameter_algebra_spec: Any,
) -> InterventionScheduleDependencyMetadata:
    """Expand direct schedule names through the admitted Algebra dependency map."""

    direct_metadata = intervention_schedule_parameter_dependencies(schedule)
    if not direct_metadata:
        return direct_metadata
    # Admission rejected dependency cycles in this graph.
    admitted_dependencies = parameter_algebra_spec.evaluation_dependencies

    expanded: dict[tuple[int, int | None, str], tuple[str, ...]] = {}
    for identity, raw_names in direct_metadata.items():
        leaves: set[str] = set()

        def visit(name: str) -> None:
            dependencies = admitted_dependencies.get(name)
            if not dependencies:
                leaves.add(name)
                return
            for dependency in dependencies:
                visit(dependency)

        for raw_name in raw_names:
            visit(raw_name)
        expanded[identity] = tuple(sorted(leaves))
    return MappingProxyType(expanded)


def _resolve_intervention_schedule_for_request(
    schedule: InterventionSchedule,
    *,
    parameter_values: Mapping[str, Any],
    dependency_metadata: InterventionScheduleDependencyMetadata | None = None,
    runtime_parameters: frozenset[str] = frozenset(),
) -> InterventionSchedule:
    try:
        return schedule.resolve_parameters(
            dict(parameter_values),
            dependency_metadata=dependency_metadata,
            runtime_parameters=runtime_parameters,
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
    parameter_algebra_spec: Any,
) -> frozenset[str]:
    if not unresolved_intervention_schedule.is_parameterized:
        return frozenset()
    requested = execution_request.requested_parameter_names
    supplied = set(dict(execution_request.parameter_overrides or {}).keys())
    referenced = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    if not requested or not referenced:
        return frozenset()

    requested_names = {
        str(name).strip()
        for name in requested
        if str(name).strip()
    }
    supplied_names = {str(name).strip() for name in supplied if str(name).strip()}
    expanded_metadata = _expanded_intervention_schedule_dependency_metadata(
        unresolved_intervention_schedule,
        parameter_algebra_spec=parameter_algebra_spec,
    )
    deferred: set[str] = set()
    for identity, direct_names in intervention_schedule_parameter_dependencies(
        unresolved_intervention_schedule
    ).items():
        if identity[2] not in {"time", "start_time", "end_time"}:
            continue
        expanded_names = set(expanded_metadata.get(identity, ()))
        if not expanded_names.intersection(requested_names - supplied_names):
            continue
        deferred.update(str(raw_name) for raw_name in direct_names)
    return frozenset(deferred)


def build_simulation_execution_request(
    *,
    prepared_dsl: PreparedDslRequestSet,
    initial_values: Iterable[tuple[str, float]],
    t_span: tuple[float, float],
    solver_config: _PreparedSolverConfig,
    parameter_values: Mapping[str, float] | None = None,
    use_authored_stops: bool = False,
) -> SimulationExecutionRequest:
    """Capture, isolate, and compile one direct Run/Preview target."""

    initials = dict(initial_values)
    species_names = prepared_dsl.analysis.species_names
    # Every value arrives admitted, one per accepted species; this capture adds
    # only generated-grid realization.
    request = SimulationExecutionRequest(
        prepared_dsl=prepared_dsl,
        species_names=species_names,
        y0=tuple(initials[name] for name in species_names),
        t_span=t_span,
        solver_config=solver_config,
        parameter_overrides=parameter_values or None,
        use_authored_stops=use_authored_stops,
    )
    try:
        # Realize the generated N grid at the Run/Preview request boundary so
        # binary64-unrepresentable spans fail before worker launch.
        build_time_grid(*request.t_span, solver_config.grid)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution grid cannot be realized on this time span.") from exc
    return request


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


def _independent_scalar_names(spec: Any) -> frozenset[str]:
    return frozenset(
        name for name in spec.parameter_dependencies
        if spec.parameter_assignment_kind(name) == "independent scalar"
    )


def _apply_parameter_overrides_to_prepared_mechanism(
    mechanism: Any,
    *,
    parameter_algebra_spec: Any,
    raw_values: Mapping[str, float],
    mechanism_parameter_names: set[str],
    scalar_parameter_names: Iterable[str],
    parameter_target_map: Mapping[str, ParameterTarget],
) -> None:
    """Apply slider-style values to one isolated execution mechanism."""
    mechanism_coordinate_values = {
        name: value
        for name, value in raw_values.items()
        if name in mechanism_parameter_names
    }
    scalar_override_values = {
        name: value
        for name, value in raw_values.items()
        if name in scalar_parameter_names
    }
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
    )

    try:
        apply_parameter_algebra_spec_to_mechanism(
            parameter_algebra_spec,
            mechanism=mechanism,
            parameter_target_map=parameter_target_map,
            mechanism_coordinate_overrides=mechanism_coordinate_values,
            scalar_parameter_overrides=scalar_override_values,
        )
    except OrdinaryParameterDomainError as exc:
        raise SimulationPreparationError(
            "ordinary_parameter_domain", str(exc), parameter_names=(exc.field,),
        ) from exc
    except Exception as exc:
        raise SimulationPreparationError(
            "parameter_overrides",
            str(exc),
            parameter_names=tuple(sorted(mechanism_coordinate_values))
            + tuple(sorted(scalar_override_values)),
        ) from exc
    return None


def prepare_simulation_run(
    *,
    execution_request: SimulationExecutionRequest,
    cancellation_control: SimulationCancellationControl | None = None,
) -> PreparedSimulationRun:
    """
    Prepare the shared scientific execution request.

    This function is the single owner for:
    - Retaining the admitted DSL temperature schedule through execution
    - Parameter algebra application
    - Generated symbolic Jacobian construction when supported by the solver path
    """

    _raise_if_cancelled(cancellation_control)
    prepared_solver_config = execution_request.solver_config
    topology = execution_request.prepared_dsl.compiled_ode_topology()
    symbolic_structure = (
        execution_request.prepared_dsl.symbolic_jacobian_structure()
        if prepared_solver_config.use_analytic_jacobian
        else None
    )
    _raise_if_cancelled(cancellation_control)

    mechanism = execution_request.prepared_dsl.mutable_mechanism_shell()
    rhs: Callable[..., np.ndarray] | None = None
    species_names = list(execution_request.species_names)
    y0 = np.asarray(execution_request.y0, dtype=float).reshape(-1).copy()
    temperature_schedule_override = execution_request.prepared_dsl.temperature_schedule_for_execution()
    parameter_algebra_spec = execution_request.prepared_dsl.parameter_algebra_spec_for_execution()
    parameter_target_map = build_parameter_target_map(mechanism)

    unresolved_intervention_schedule = (
        _fresh_preparation_unresolved_intervention_schedule_for_request(
            execution_request
        )
    )
    schedule_only_values = _bind_request_parameter_overrides(
        mechanism=mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
        parameter_overrides=execution_request.parameter_overrides,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        parameter_target_map=parameter_target_map,
    )
    _raise_if_cancelled(cancellation_control)

    try:
        _coefficient_evaluator, rhs = topology.bind_target(
            mechanism, base_values=schedule_only_values
        )
    except Exception as exc:
        raise SimulationPreparationError("ode_build", str(exc)) from exc

    intervention_execution = execution_request.prepared_dsl.intervention_execution_contract()
    if intervention_execution is not None and not execution_request.use_authored_stops:
        intervention_execution = intervention_execution.without_stops()
    if unresolved_intervention_schedule is not None:
        if _schedule_parameters_deferred_to_prepared_request(
            execution_request,
            unresolved_intervention_schedule,
            parameter_algebra_spec=parameter_algebra_spec,
        ):
            unresolved_intervention_schedule.validate_species(species_names)
        else:
            try:
                dependency_metadata = _expanded_intervention_schedule_dependency_metadata(
                    unresolved_intervention_schedule,
                    parameter_algebra_spec=parameter_algebra_spec,
                )
                direct_dependencies = intervention_schedule_parameter_dependencies(unresolved_intervention_schedule)
                clock_fields = {"time", "start_time", "end_time"}
                clock_roots = tuple(sorted({
                    name for identity, names in direct_dependencies.items()
                    if identity[2] in clock_fields for name in names
                }))
                payload_names = frozenset(
                    name for identity, names in direct_dependencies.items()
                    if identity[2] not in clock_fields for name in names
                )
                resolved_schedule = _resolve_intervention_schedule_for_request(
                    unresolved_intervention_schedule,
                    parameter_values=(_coefficient_evaluator.parameter_values(
                        _coefficient_evaluator.nominal_temperature_K, clock_roots
                    ) if clock_roots else {}),
                    runtime_parameters=payload_names,
                    dependency_metadata=dependency_metadata,
                )
                resolved_schedule.validate_species(species_names)
                intervention_execution = compile_intervention_schedule(
                    resolved_schedule,
                    authored_payload=unresolved_intervention_schedule.to_payload(),
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

    _raise_if_cancelled(cancellation_control)

    try:
        temperature_schedule, runtime_symbolic_jacobian = build_prepared_run_context(
            mechanism=mechanism,
            temperature_schedule=temperature_schedule_override,
            use_analytic_jacobian=prepared_solver_config.use_analytic_jacobian,
            symbolic_structure=symbolic_structure,
            coefficient_evaluator=_coefficient_evaluator,
        )
    except Exception as exc:
        raise SimulationPreparationError("analytic_jacobian", str(exc)) from exc
    _raise_if_cancelled(cancellation_control)

    initials_for_algebra = {sp: float(y0[idx]) for idx, sp in enumerate(species_names)}

    from kindred.core.simulation_algebra_context import runtime_algebra_symbol_names
    from kindred.core.temperature import temperature_at_execution_time
    runtime_names = runtime_algebra_symbol_names(parameter_algebra_spec)
    symbols = _coefficient_evaluator.parameter_values
    stop_observable_value = None
    if intervention_execution is not None:
        from kindred.core.intervention_schedule import ObservableTriggerCondition, InterventionStopAction
        stops = tuple(
            item for item in intervention_execution.triggers
            if isinstance(item.then, InterventionStopAction) and isinstance(item.condition, ObservableTriggerCondition)
        )
        targets = tuple(dict.fromkeys(item.condition.name for item in stops))
        if targets:
            from kindred.core.algebra.errors import AlgebraError
            from kindred.core.algebra.simulation_series import compiled_algebra_for_mechanism, prepare_algebra_stop_evaluator
            try:
                compiled = compiled_algebra_for_mechanism(mechanism)
                target_roots = {
                    name: tuple(root for root in compiled.parameter_roots(name)
                                if root in parameter_algebra_spec.evaluation_dependencies)
                    for name in targets
                }
                static_roots = tuple(sorted({
                    root for roots in target_roots.values() for root in roots
                    if root not in runtime_names
                }))
                static = (symbols(temperature_at_execution_time(
                    temperature_schedule, _coefficient_evaluator.nominal_temperature_K,
                    execution_request.t_span[0],
                ), static_roots) if static_roots else {})
                evaluator = prepare_algebra_stop_evaluator(
                    compiled,
                    species_names=tuple(species_names), initials=initials_for_algebra,
                    symbol_values=static, runtime_symbol_names=set(runtime_names),
                )
                for target in targets:
                    evaluator.admit(target)
            except Exception as exc:
                # The refusal states what failed on the line it read.
                line, text = (exc.line, exc.line_text) if isinstance(exc, AlgebraError) else (None, None)
                raise SimulationPreparationError(
                    "stopping_intervention", located(f"A stop condition cannot watch this value: {exc}", line, text),
                ) from exc

            def stop_observable_value(name: str, time: float, state: np.ndarray, temperature_K: float) -> float:
                return evaluator.evaluate(name, time, state, symbols(temperature_K, target_roots[name]))

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
        request=request,
        unresolved_intervention_schedule=unresolved_intervention_schedule,
        cancellation_control=cancellation_control,
        stop_observable_value=stop_observable_value,
        parameter_values_at_temperature=symbols,
    )


def _bind_request_parameter_overrides(
    *,
    mechanism: Any,
    parameter_algebra_spec: Any,
    parameter_overrides: Mapping[str, float] | None,
    unresolved_intervention_schedule: Any,
    parameter_target_map: Mapping[str, ParameterTarget],
) -> Mapping[str, float]:
    """Apply owned coordinates and return admitted schedule-only bindings.

    Override names and values arrive admitted by Slider, Scan or Fit as
    enumerated independent coordinates and scalars of this accepted mechanism.
    """

    declared_scalar_names = _independent_scalar_names(parameter_algebra_spec)
    raw_values = dict(parameter_overrides or {})
    schedule_names = set(intervention_schedule_parameter_names(unresolved_intervention_schedule))
    mechanism_names = {name for name in raw_values if name in parameter_target_map}
    scalar_names = {
        name for name in raw_values
        if name not in mechanism_names and name in declared_scalar_names
    }
    _apply_parameter_overrides_to_prepared_mechanism(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
        raw_values=raw_values,
        mechanism_parameter_names=mechanism_names,
        scalar_parameter_names=scalar_names,
        parameter_target_map=parameter_target_map,
    )
    return {
        name: value
        for name, value in raw_values.items()
        if name in schedule_names and name not in mechanism_names | scalar_names
    }
