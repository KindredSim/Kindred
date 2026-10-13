from __future__ import annotations

from kindred.core.quantity_units import CUBIC_METRE, MOLAR, SECOND, Unit, presentation_unit

from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from kindred.core.exceptions import SimulationCancelled
from kindred.core.simulation_algebra_context import runtime_algebra_symbol_names
from kindred.core.simulation_failure import FactualCause
from kindred.core.algebra.errors import UNCOMPUTABLE_ERRORS
from kindred.core.algebra.evaluator import AlgebraObservableError
from kindred.core.result_repository import (
    SimulationResultValue,
)
from kindred.core.simulator.solver_types import FinalSimulationOutput, SimulationCancellationControl
from kindred.core.algebra.simulation_series import (
    evaluate_algebra_block_for_simulation_partial,
    observable_parameter_roots,
    observable_reads_species,
)

__all__ = [
    "finalize_simulation_result",
    "result_quantity_units",
]


def result_quantity_units(
    parameter_algebra_spec: object,
    species_names: tuple[str, ...] | list[str],
) -> tuple[dict[str, Unit | None], dict[str, Unit | None]]:
    """Scientific units (species in M, t in s) and their presentation units."""

    units = dict(parameter_algebra_spec.quantity_units)
    units.update({name: MOLAR for name in species_names})
    units["t"] = SECOND
    declarations = {
        line.name: line.declared_unit
        for line in parameter_algebra_spec.observable_block.lines
    }
    return units, {
        name: presentation_unit(unit, declarations.get(name))
        for name, unit in units.items()
    }


def _freeze_solver_provenance_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(name): _freeze_solver_provenance_value(item)
                for name, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_solver_provenance_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            _freeze_solver_provenance_value(item)
            for item in sorted(value, key=repr)
        )
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


def _freeze_solver_provenance_mapping(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(name): _freeze_solver_provenance_value(item)
            for name, item in value.items()
        }
    )




def _algebra_cause(error: AlgebraObservableError, spec: object) -> FactualCause:
    """Project one expected per-observable Algebra failure.

    A let its own formula cannot compute records the independent inputs it
    reads, as a param's failure does, so a Fit decides from them."""

    details: dict[str, object] = {**error.details, "name": error.name}
    message = error.message
    if "undefined_points" in details:
        # A let undefined at some time points only names where.
        message = (f"{message} Undefined at {details['undefined_points']} of {details['points']} "
                   f"time points, first at t = {details['first_undefined_time']:g} s.")
    if error.exc_type in {error_type.__name__ for error_type in UNCOMPUTABLE_ERRORS}:
        details["stage"] = "result_algebra"
        details["parameter_names"] = spec.input_dependencies(
            observable_parameter_roots(spec.observable_block, error.name)
        )
        details["reads_species"] = observable_reads_species(spec.observable_block, error.name)
    if error.col is not None:
        details["col"] = error.col
    # The authored statement is the cause's location, as for every other cause.
    context = {key: getattr(error, key) for key in ("line", "line_text")
               if getattr(error, key) is not None}
    return FactualCause(
        domain="simulation",
        origin="algebra",
        code=error.code,
        type=error.exc_type,
        message=message,
        context=context,
        details=details,
    )


def _freeze_algebra_series(
    series: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Freeze evaluator-owned series exactly once before publication."""

    frozen: dict[str, np.ndarray] = {}
    for raw_name, raw_values in series.items():
        values = np.asarray(raw_values, dtype=float)
        values.setflags(write=False)
        frozen[str(raw_name)] = values
    return frozen


def finalize_simulation_result(
    *,
    result: FinalSimulationOutput,
    species_names: list[str],
    initials_for_algebra: Mapping[str, float],
    parameter_values_at_temperature: Callable[[float, tuple[str, ...]], Mapping[str, float]],
    parameter_algebra_spec: object,
    parameters: tuple[tuple[str, float, str], ...] = (),
    parameter_classifications: tuple[tuple[str, str, str], ...] = (),
    cancellation_control: SimulationCancellationControl | None = None,
) -> SimulationResultValue:
    """Admit one solver output through canonical row finalization.

    The solver's direct identity arrays and actual executed-segment catalog are
    transported unchanged into :class:`SimulationResultValue`; this boundary
    never repairs or reconstructs row identity.
    """

    def check_cancelled() -> None:
        if cancellation_control is not None and cancellation_control.cancel_requested():
            raise SimulationCancelled()

    check_cancelled()
    result_time = result.t
    result_y = result.Y
    # The accepted mechanism's unique species name the solver's state rows.
    base_species_names = list(species_names)

    finalized_series: dict[str, np.ndarray] = {
        name: result_y[index, :]
        for index, name in enumerate(base_species_names)
    }
    finalized_algebra_scalars: dict[str, float] = {}
    finalized_algebra_errors: tuple[FactualCause, ...] = ()
    observable_block = parameter_algebra_spec.observable_block
    if observable_block.lines:
        species_series = {
            name: result_y[index, :]
            for index, name in enumerate(base_species_names)
        }
        runtime_names = runtime_algebra_symbol_names(parameter_algebra_spec)
        symbol_failures: dict[str, Exception] = {}

        def resolve_symbol(name: str) -> float | np.ndarray | None:
            check_cancelled()
            if name not in parameter_algebra_spec.evaluation_dependencies:
                return None
            if name in symbol_failures:
                raise symbol_failures[name]
            try:
                if name in runtime_names:
                    values = np.empty(result.temperature_K.size, dtype=float)
                    for index, temperature in enumerate(result.temperature_K):
                        if index % 256 == 0:
                            check_cancelled()
                        values[index] = parameter_values_at_temperature(float(temperature), (name,))[name]
                    check_cancelled()
                    return values
                return parameter_values_at_temperature(float(result.temperature_K[0]), (name,))[name]
            except SimulationCancelled:
                raise
            except Exception as exc:
                symbol_failures[name] = exc
                raise
        algebra_series, algebra_scalars, algebra_errors = (
            evaluate_algebra_block_for_simulation_partial(
                observable_block,
                t=result_time,
                species_series=species_series,
                initials=initials_for_algebra,
                symbol_series={},
                symbol_values={},
                symbol_resolver=resolve_symbol,
                runtime_symbol_names=runtime_names,
                cancellation_check=None if cancellation_control is None else cancellation_control.cancel_requested,
            )
        )
        finalized_algebra_errors = tuple(
            _algebra_cause(error, parameter_algebra_spec) for error in algebra_errors
        )
        finalized_series.update(_freeze_algebra_series(algebra_series))
        finalized_algebra_scalars = dict(algebra_scalars)

    frozen_solver_provenance = _freeze_solver_provenance_mapping(result.provenance)
    requested_algebra_outputs = tuple(statement.name for statement in observable_block.lines)
    succeeded_names = set(finalized_algebra_scalars)
    succeeded_names.update(
        name for name in finalized_series if name not in base_species_names
    )
    failed_names = {
        str(error.details.get("name") or "")
        for error in finalized_algebra_errors
        if str(error.details.get("name") or "") in requested_algebra_outputs
    }
    succeeded_algebra_outputs = tuple(
        name
        for name in requested_algebra_outputs
        if name in succeeded_names
    )
    failed_algebra_outputs = tuple(
        name
        for name in requested_algebra_outputs
        if name in failed_names and name not in succeeded_names
    )
    check_cancelled()
    units, preferred_units = result_quantity_units(parameter_algebra_spec, species_names)
    names = ("t", *finalized_series, *finalized_algebra_scalars)
    presentation_units = tuple((name, preferred_units.get(name)) for name in names)
    if result.volume is not None:
        initializer, _identity = result.intervention_execution.initial_volume
        declared = initializer.declared_unit
        if initializer.parameter is not None:
            parameter_units = {s.name: s.declared_unit for s in parameter_algebra_spec.parameter_block.lines}
            parameter_unit = units.get(initializer.parameter)
            declared = parameter_units.get(initializer.parameter) if parameter_unit is not None else None
        presentation_units += (("Vessel volume", presentation_unit(CUBIC_METRE, declared)),)
    return SimulationResultValue(
        temperature_K=result.temperature_K,
        quantity_units=tuple((name, units.get(name)) for name in names),
        presentation_units=presentation_units,
        t=result_time,
        series=MappingProxyType(finalized_series),
        segment_ids=result.segment_ids,
        segment_keys=result.segment_keys,
        discontinuity_ids=result.discontinuity_ids,
        scalars=MappingProxyType(finalized_algebra_scalars),
        parameters=parameters,
        parameter_classifications=parameter_classifications,
        solver_provenance=frozen_solver_provenance,
        requested_algebra_outputs=requested_algebra_outputs,
        succeeded_algebra_outputs=succeeded_algebra_outputs,
        failed_algebra_outputs=failed_algebra_outputs,
        algebra_errors=finalized_algebra_errors,
        intervention_execution=getattr(result, "intervention_execution"),
        presentation_boundaries=result.presentation_boundaries,
        volume=result.volume,
        termination=result.termination,
    )
