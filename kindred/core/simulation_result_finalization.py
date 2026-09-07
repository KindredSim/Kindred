from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from kindred.core.simulation_algebra_context import prepare_simulation_algebra_symbols
from kindred.core.simulation_failure import FactualCause
from kindred.core.result_repository import (
    SimulationResultValue,
)
from kindred.core.simulator.solver_types import FinalSimulationOutput
from kindred.core.temperature import (
    TemperatureScheduleProtocol,
)
from kindred.core.algebra.simulation_series import (
    compiled_algebra_for_mechanism,
    evaluate_compiled_algebra_series_for_simulation_partial,
)

__all__ = [
    "build_simulation_row_algebra_context",
    "finalize_simulation_result",
]


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

def _initials_for_algebra(
    mechanism: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
) -> dict[str, float]:
    if isinstance(initials_for_algebra, Mapping):
        return {str(name): float(value) for name, value in dict(initials_for_algebra).items()}
    species = getattr(mechanism, "species", {}) or {}
    return {
        str(name): float(getattr(species[str(name)], "initial_conc"))
        for name in species_names
        if str(name) in species
    }


def build_simulation_row_algebra_context(
    mechanism: object,
    times: np.ndarray,
    *,
    parameter_algebra_spec: object,
    temperature_schedule: TemperatureScheduleProtocol | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read row-local temperature, coefficients, and prepared static scalars."""
    static, names, symbols = prepare_simulation_algebra_symbols(
        mechanism, parameter_algebra_spec=parameter_algebra_spec,
        temperature_schedule=temperature_schedule,
    )
    row_times = np.asarray(times, dtype=float)
    if row_times.ndim != 1 or not row_times.size or np.any(~np.isfinite(row_times)) or np.any(np.diff(row_times) <= 0.0):
        raise ValueError("result row times must be finite and strictly increasing.")
    rows = [symbols(float(time)) for time in row_times] if temperature_schedule is not None else [symbols(float(row_times[0]))] * len(row_times)
    return {name: np.asarray([row[name] for row in rows], dtype=float) for name in names}, static


def _algebra_cause(error: object) -> FactualCause:
    """Project one expected per-observable Algebra failure."""

    details: dict[str, object] = {}
    name = getattr(error, "name", None)
    if name is not None:
        details["name"] = str(name)
    for source_name in ("line", "col", "line_text"):
        value = getattr(error, source_name, None)
        if value is not None:
            details[source_name] = value
    code = getattr(error, "code", None)
    return FactualCause(
        domain="simulation",
        origin="algebra",
        code=str(code) if code is not None else None,
        type=str(getattr(error, "exc_type", None) or type(error).__name__),
        message=str(getattr(error, "message", None) or str(error)),
        details=details,
    )


def _freeze_algebra_series(
    series: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Freeze evaluator-owned series exactly once before publication."""

    frozen: dict[str, np.ndarray] = {}
    for raw_name, raw_values in series.items():
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1:
            values = values.reshape(-1)
        values.setflags(write=False)
        frozen[str(raw_name)] = values
    return frozen


def finalize_simulation_result(
    *,
    mechanism: object,
    result: FinalSimulationOutput,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
    temperature_schedule: TemperatureScheduleProtocol | None,
    parameter_algebra_spec: object,
    parameters: tuple[tuple[str, float, str], ...] = (),
    solver_provenance_updates: Mapping[str, object] | None = None,
) -> SimulationResultValue:
    """Admit one solver output through canonical row finalization.

    The solver's direct identity arrays and actual executed-segment catalog are
    transported unchanged into :class:`SimulationResultValue`; this boundary
    never repairs or reconstructs row identity.
    """

    result_time = result.t
    result_y = result.Y
    base_species_names = [str(name) for name in species_names]
    if (
        not base_species_names
        or any(not name.strip() for name in base_species_names)
        or len(set(base_species_names)) != len(base_species_names)
    ):
        raise ValueError("base simulation species names must be nonblank and unique.")
    if result_y.shape[0] != len(base_species_names):
        raise ValueError(
            "solver result matrix must align with base species and row time."
        )

    finalized_series: dict[str, np.ndarray] = {
        name: result_y[index, :]
        for index, name in enumerate(base_species_names)
    }
    finalized_algebra_scalars: dict[str, float] = {}
    finalized_algebra_errors: tuple[FactualCause, ...] = ()
    compiled_algebra = compiled_algebra_for_mechanism(mechanism)
    if compiled_algebra is not None:
        species_series = {
            name: result_y[index, :]
            for index, name in enumerate(base_species_names)
        }
        symbol_series, symbol_values = build_simulation_row_algebra_context(
            mechanism,
            result_time,
            temperature_schedule=temperature_schedule,
            parameter_algebra_spec=parameter_algebra_spec,
        )
        initials = _initials_for_algebra(
            mechanism,
            base_species_names,
            initials_for_algebra,
        )
        algebra_series, algebra_scalars, algebra_errors = (
            evaluate_compiled_algebra_series_for_simulation_partial(
                compiled_algebra,
                t=result_time,
                species_series=species_series,
                initials=initials,
                symbol_series=symbol_series,
                symbol_values=symbol_values,
            )
        )
        finalized_algebra_errors = tuple(
            _algebra_cause(error) for error in algebra_errors
        )
        finalized_series.update(_freeze_algebra_series(algebra_series))
        finalized_algebra_scalars = dict(algebra_scalars)

    completed_solver_provenance = dict(result.provenance)
    if solver_provenance_updates:
        completed_solver_provenance.update(solver_provenance_updates)
    frozen_solver_provenance = _freeze_solver_provenance_mapping(
        completed_solver_provenance
    )
    requested_algebra_outputs = (
        ()
        if compiled_algebra is None
        else tuple(compiled_algebra.observable_names)
    )
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
    omitted_algebra_outputs = tuple(
        name
        for name in requested_algebra_outputs
        if name not in succeeded_names and name not in failed_names
    )
    return SimulationResultValue(
        t=result_time,
        series=finalized_series,
        segment_ids=result.segment_ids,
        segment_keys=result.segment_keys,
        discontinuity_ids=result.discontinuity_ids,
        scalars=finalized_algebra_scalars,
        parameters=parameters,
        solver_provenance=frozen_solver_provenance,
        requested_algebra_outputs=requested_algebra_outputs,
        succeeded_algebra_outputs=succeeded_algebra_outputs,
        failed_algebra_outputs=failed_algebra_outputs,
        omitted_algebra_outputs=omitted_algebra_outputs,
        algebra_errors=finalized_algebra_errors,
        intervention_execution=getattr(result, "intervention_execution"),
        presentation_boundaries=result.presentation_boundaries,
        termination=result.termination,
    )
