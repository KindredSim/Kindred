"""Shared scientific execution for one admitted Run or Preview request."""
from __future__ import annotations

from kindred.core.exceptions import SimulationCancelled
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulation_parameters import enumerate_simulation_parameters
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest, SimulationPreparationError, prepare_simulation_run,
    build_simulation_request_from_prepared_run,
)
from kindred.core.simulation_result_finalization import finalize_simulation_result
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulator.solvers import solve_ode


def execute_simulation(
    execution_request: SimulationExecutionRequest,
    *,
    cancellation_control: SimulationCancellationControl | None = None,
) -> SimulationResultValue:
    """Prepare, solve, and finalize without application or GUI ownership."""
    def check_cancelled() -> None:
        if cancellation_control is not None and cancellation_control.cancel_requested():
            raise SimulationCancelled()

    check_cancelled()
    coordinates = enumerate_simulation_parameters(execution_request.prepared_dsl)
    by_name = {item.name: item for item in coordinates}
    overrides = dict(execution_request.parameter_overrides or {})
    for name in overrides:
        coordinate = by_name.get(name)
        if coordinate is None:
            raise ValueError(f"Preview parameter {name!r} is not a direct GUI coordinate.")
        if not coordinate.writable or coordinate.derived:
            raise ValueError(f"Preview parameter {name!r} is owned by retained Algebra.")
    parameters = tuple((item.name, overrides.get(item.name, item.value), item.unit) for item in coordinates)
    prepared = prepare_simulation_run(execution_request=execution_request, cancellation_control=cancellation_control)
    if prepared.intervention_execution is None:
        raise SimulationPreparationError("intervention_schedule", "Prepared Run retained an unresolved intervention schedule.")
    solved = solve_ode(build_simulation_request_from_prepared_run(
        prepared, y0=prepared.y0, intervention_execution=prepared.intervention_execution,
        symbolic_jacobian=prepared.symbolic_jacobian, cancellation_control=cancellation_control,
    ))
    check_cancelled()
    result = finalize_simulation_result(
        mechanism=prepared.mechanism, result=solved, species_names=list(prepared.species_names),
        initials_for_algebra=prepared.initials_for_algebra, parameter_values_at_time=prepared.parameter_values_at_time,
        parameter_algebra_spec=prepared.parameter_algebra_spec, parameters=parameters,
    )
    check_cancelled()
    return result
