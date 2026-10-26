"""Shared scientific execution for one admitted Run or Preview request."""
from __future__ import annotations

from kindred.core.exceptions import SimulationCancelled
from kindred.core.quantity_units import convert_value
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulation_parameters import derived_parameter_values, enumerate_simulation_parameters
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest, prepare_simulation_run,
    build_simulation_request_from_prepared_run,
)
from kindred.core.simulation_result_finalization import finalize_simulation_result
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulator.solvers import solve_ode


def execute_simulation(
    execution_request: SimulationExecutionRequest,
    *,
    cancellation_control: SimulationCancellationControl | None = None,
    species_statistics: bool = False,
) -> SimulationResultValue:
    """Prepare, solve, and finalize without application or GUI ownership.

    ``species_statistics`` asks for each species' maximum and its time, and its
    cumulative temporal concentration, which a Run with interventions other
    than temperature changes never carries.
    """
    def check_cancelled() -> None:
        if cancellation_control is not None and cancellation_control.cancel_requested():
            raise SimulationCancelled()

    check_cancelled()
    coordinates = enumerate_simulation_parameters(execution_request.prepared_dsl)
    # Preview and Scan admitted their override names as these coordinates.
    overrides = dict(execution_request.parameter_overrides or {})
    spec = execution_request.prepared_dsl.parameter_algebra_spec_for_execution()
    inputs, quantities = dict(spec.input_units), dict(spec.quantity_units)
    parameters = tuple((item.name, convert_value(overrides.get(item.name, item.value), inputs[item.name], quantities[item.name]), quantities[item.name].symbol) for item in coordinates)
    # Every prepared run carries a compiled schedule, an empty one included.
    prepared = prepare_simulation_run(execution_request=execution_request, cancellation_control=cancellation_control)
    solved = solve_ode(build_simulation_request_from_prepared_run(
        prepared,
        species_ctc=species_statistics and prepared.intervention_execution.is_empty,
        species_maximum=species_statistics,
    ))
    check_cancelled()
    derived = derived_parameter_values(
        execution_request.prepared_dsl, prepared.parameter_values_at_temperature, solved,
    )
    result = finalize_simulation_result(
        result=solved, species_names=list(prepared.species_names),
        initials_for_algebra=prepared.initials_for_algebra, parameter_values_at_temperature=prepared.parameter_values_at_temperature,
        parameter_algebra_spec=prepared.parameter_algebra_spec, parameters=parameters,
        parameter_classifications=tuple(
            (item.name, item.family, item.role) for item in coordinates
        ),
        derived_parameters=derived,
        cancellation_control=cancellation_control,
    )
    check_cancelled()
    return result
