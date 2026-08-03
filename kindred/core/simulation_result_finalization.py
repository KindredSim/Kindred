from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from kindred.core import simulation_algebra_policy
from kindred.core.equilibrium_rate_authority import effective_equilibrium_keq
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator
from kindred.core.simulation_failure import build_simulation_failure
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.result_repository import (
    SimulationResultProvenance,
    SimulationResultValue,
)
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.step_indexing import iter_canonical_parameters
from kindred.core.temperature import (
    TemperatureScheduleProtocol,
    coerce_temperature_schedule,
)
from kindred.core.algebra.simulation_series import (
    compiled_algebra_for_mechanism,
    evaluate_compiled_algebra_series_for_simulation,
    evaluate_compiled_algebra_series_for_simulation_partial,
)
from kindred.core.simulation_result_payload import (
    build_secondary_simulation_success_payload,
    build_simulation_success_payload,
)

__all__ = [
    "FinalizedSimulationResult",
    "build_finalized_simulation_result_payload",
    "build_simulation_row_algebra_context",
    "evaluate_simulation_result_algebra",
    "finalize_simulation_result",
]

@dataclass(frozen=True)
class FinalizedSimulationResult:
    y: np.ndarray
    species_names: list[str]
    base_species_count: int
    algebra_scalars: dict[str, float]
    algebra_errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", self.y.copy())
        object.__setattr__(self, "species_names", list(self.species_names))
        object.__setattr__(self, "algebra_scalars", deepcopy(dict(self.algebra_scalars)))
        object.__setattr__(self, "algebra_errors", [deepcopy(dict(error)) for error in self.algebra_errors])
        object.__setattr__(self, "warnings", [deepcopy(dict(warning)) for warning in self.warnings])


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


def _prepared_static_scalar_snapshot(mechanism: object) -> dict[str, float]:
    """Read the complete static scalar map already bound during preparation."""

    metadata = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("prepared mechanism metadata must be a mapping.")
    parameter_spec = metadata.get("parameter_algebra_spec")
    if parameter_spec is None:
        return {}
    assignment_names = getattr(parameter_spec, "param_assignment_names", None)
    if not callable(assignment_names):
        raise TypeError(
            "prepared parameter Algebra spec must expose assignment names."
        )
    mechanism_names = {
        str(name)
        for name in getattr(parameter_spec, "mechanism_param_names", ()) or ()
    }
    scalar_names = tuple(
        sorted(
            str(name)
            for name in assignment_names()
            if str(name) not in mechanism_names
        )
    )
    raw_values = metadata.get("scalar_params")
    scalar_values = raw_values if isinstance(raw_values, Mapping) else {}
    raw_bindings = metadata.get("scalar_param_bindings")
    scalar_bindings = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    values: dict[str, float] = {}
    for name in scalar_names:
        if name in scalar_bindings:
            binding = scalar_bindings[name]
            raw_value = binding() if callable(binding) else binding
        elif name in scalar_values:
            raw_value = scalar_values[name]
        else:
            raise ValueError(
                f"prepared static scalar snapshot is missing {name!r}."
            )
        if type(raw_value) is bool:
            raise TypeError(
                f"prepared static scalar {name!r} must be numeric."
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"prepared static scalar {name!r} must be numeric."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"prepared static scalar {name!r} must be finite."
            )
        values[name] = value
    return values


def build_simulation_row_algebra_context(
    mechanism: object,
    times: np.ndarray,
    *,
    temperature_schedule: TemperatureScheduleProtocol | Mapping[str, object] | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build the sole row-local temperature/coefficient/static context.

    ``mechanism`` is the already prepared mechanism. Static ``param`` values
    are read, never evaluated, at this boundary.
    """

    metadata = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("prepared mechanism metadata must be a mapping.")
    mechanism_metadata = MechanismMetadataView.from_metadata(metadata)
    nominal_temperature = float(mechanism_metadata.temperature_K)
    schedule = coerce_temperature_schedule(temperature_schedule)
    row_times = np.asarray(times, dtype=float)
    if row_times.ndim != 1 or row_times.size == 0:
        raise ValueError("result row times must be a nonempty 1D array.")
    if np.any(~np.isfinite(row_times)) or np.any(np.diff(row_times) <= 0.0):
        raise ValueError("result row times must be finite and strictly increasing.")
    temperatures = np.asarray(
        [
            float(schedule(float(time))) if schedule is not None else nominal_temperature
            for time in row_times
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError(
            "row temperature authority must be positive and finite."
        )
    coefficient_evaluator = build_effective_rate_coefficient_evaluator(mechanism)
    reactions = list(getattr(mechanism, "reactions", ()) or ())
    equilibria = list(getattr(mechanism, "equilibria", ()) or ())
    namespace = build_namespace_from_mechanism(mechanism)
    canonical_names = {
        str(item.canonical_name) for item in namespace.ordered_items
    }
    coefficient_rows: dict[str, list[float]] = {}
    for temperature in temperatures:
        coefficients = coefficient_evaluator(float(temperature))
        for name, entry, role in iter_canonical_parameters(mechanism):
            kind = str(entry.get("kind") or "")
            if kind == "reaction" and role == "k":
                index = int(entry.get("reaction_index", -1))
                value = coefficients.forward[index]
            elif kind == "equilibrium":
                index = int(entry.get("equilibrium_index", -1))
                coefficient_index = len(reactions) + index
                if role == "kf":
                    value = coefficients.forward[coefficient_index]
                elif role == "kr":
                    value = coefficients.reverse[coefficient_index]
                elif role == "Keq":
                    value = effective_equilibrium_keq(
                        equilibria[index], temperature_K=float(temperature)
                    )
                else:
                    continue
            else:
                continue
            if value is None:
                raise ValueError(
                    f"chemistry coefficient readout {name!r} is unavailable."
                )
            canonical_name = str(name)
            if canonical_name not in canonical_names:
                raise ValueError(
                    f"chemistry coefficient {canonical_name!r} is outside the admitted namespace."
                )
            coefficient_rows.setdefault(canonical_name, []).append(float(value))
    symbol_series = {"T": temperatures}
    symbol_series.update(
        {
            name: np.asarray(values, dtype=float)
            for name, values in coefficient_rows.items()
            if len(values) == temperatures.size
        }
    )
    return symbol_series, _prepared_static_scalar_snapshot(mechanism)


def evaluate_simulation_result_algebra(
    *,
    mechanism: object,
    result: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
    temperature_schedule: TemperatureScheduleProtocol | Mapping[str, object] | None,
    simulation_plan: SimulationPlan | Mapping[str, Any] | None,
) -> FinalizedSimulationResult:
    base_species_names = [str(name) for name in species_names]
    if (
        not base_species_names
        or any(not name.strip() for name in base_species_names)
        or len(set(base_species_names)) != len(base_species_names)
    ):
        raise ValueError("base simulation species names must be nonblank and unique.")
    result_y = np.asarray(getattr(result, "Y"), dtype=float)
    result_t = np.asarray(getattr(result, "t"), dtype=float)
    if result_t.ndim != 1 or result_t.size == 0:
        raise ValueError("solver result time must be a nonempty 1D array.")
    if np.any(~np.isfinite(result_t)) or np.any(np.diff(result_t) <= 0.0):
        raise ValueError("solver result time must be finite and strictly increasing.")
    if (
        result_y.ndim != 2
        or result_y.shape != (len(base_species_names), result_t.size)
        or np.any(~np.isfinite(result_y))
    ):
        raise ValueError(
            "solver result matrix must align with base species and row time."
        )
    algebra_scalars: dict[str, float] = {}
    algebra_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if compiled_algebra_for_mechanism(mechanism) is None:
        return FinalizedSimulationResult(
            y=result_y,
            species_names=base_species_names,
            base_species_count=len(base_species_names),
            algebra_scalars=algebra_scalars,
            algebra_errors=algebra_errors,
            warnings=warnings,
        )

    species_series = {
        str(species): result_y[index, :]
        for index, species in enumerate(base_species_names)
    }
    compiled = compiled_algebra_for_mechanism(mechanism)
    if compiled is None:
        return FinalizedSimulationResult(
            y=result_y,
            species_names=base_species_names,
            base_species_count=len(base_species_names),
            algebra_scalars=algebra_scalars,
            algebra_errors=algebra_errors,
            warnings=warnings,
        )
    policy = simulation_algebra_policy.algebra_policy_from_simulation_plan(
        simulation_plan,
        default=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )
    authority_error: Exception | None = None
    try:
        symbol_series, symbol_values = build_simulation_row_algebra_context(
            mechanism,
            result_t,
            temperature_schedule=temperature_schedule,
        )
    except Exception as exc:
        # Keep authority/schedule failures inside the existing policy boundary:
        # GUI and batch remain best-effort while fitting remains strict.
        authority_error = exc
        symbol_series, symbol_values = {}, {}

    def _best_effort_evaluator(
        _mechanism: object,
        *,
        t: np.ndarray,
        species_series: Mapping[str, np.ndarray],
        initials: Mapping[str, float],
    ):
        if authority_error is not None:
            raise authority_error
        return evaluate_compiled_algebra_series_for_simulation_partial(
            _mechanism,
            compiled,
            t=t,
            species_series=dict(species_series),
            initials=dict(initials),
            symbol_series=symbol_series,
            symbol_values=symbol_values,
        )

    def _strict_evaluator(
        _mechanism: object,
        *,
        t: np.ndarray,
        species_series: Mapping[str, np.ndarray],
        initials: Mapping[str, float],
    ):
        if authority_error is not None:
            raise authority_error
        return evaluate_compiled_algebra_series_for_simulation(
            _mechanism,
            compiled,
            t=t,
            species_series=dict(species_series),
            initials=dict(initials),
            symbol_series=symbol_series,
            symbol_values=symbol_values,
        )

    initials = _initials_for_algebra(
        mechanism,
        base_species_names,
        initials_for_algebra,
    )
    if policy is SimulationAlgebraPolicy.FITTING_STRICT:
        strict_series, strict_scalars = _strict_evaluator(
            mechanism,
            t=result_t,
            species_series=species_series,
            initials=initials,
        )
        evaluation = simulation_algebra_policy.SimulationAlgebraEvaluation(
            series=strict_series,
            scalars=strict_scalars,
            errors=[],
            warning=None,
        )
    else:
        evaluation = simulation_algebra_policy.evaluate_simulation_algebra(
            policy,
            mechanism,
            t=result_t,
            species_series=species_series,
            initials=initials,
            gui_evaluator=_best_effort_evaluator,
            batch_evaluator=_strict_evaluator,
        )
    algebra_scalars = dict(evaluation.scalars)
    algebra_errors = list(evaluation.errors)
    if evaluation.warning is not None:
        warnings.append(dict(evaluation.warning))
    if not evaluation.series:
        return FinalizedSimulationResult(
            y=result_y,
            species_names=base_species_names,
            base_species_count=len(base_species_names),
            algebra_scalars=algebra_scalars,
            algebra_errors=algebra_errors,
            warnings=warnings,
        )

    algebra_names = [str(name) for name in evaluation.series.keys()]
    algebra_matrix = np.vstack([np.asarray(evaluation.series[name], dtype=float) for name in algebra_names])
    return FinalizedSimulationResult(
        y=np.vstack([result_y, algebra_matrix]),
        species_names=base_species_names + algebra_names,
        base_species_count=len(base_species_names),
        algebra_scalars=algebra_scalars,
        algebra_errors=algebra_errors,
        warnings=warnings,
    )


def _required_direct_row_identity(
    result: object,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
]:
    provenance = getattr(result, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise TypeError("solver result provenance must be a mapping.")
    fields = (
        "result_segment_ids",
        "result_segment_keys",
        "result_discontinuity_ids",
    )
    missing = tuple(field for field in fields if field not in provenance)
    if missing:
        raise ValueError(
            "solver result is missing direct row identity: "
            + ", ".join(missing)
        )
    retired = frozenset(
        {
            "boundary_sides",
            "boundary_phases",
            "result_boundary_sides",
            "result_boundary_phases",
        }
    ).intersection(provenance)
    if retired:
        raise ValueError(
            "solver result contains retired boundary identity: "
            + ", ".join(sorted(retired))
        )
    return (
        np.asarray(provenance["result_segment_ids"]),
        tuple(str(value) for value in provenance["result_segment_keys"]),
        np.asarray(provenance["result_discontinuity_ids"]),
    )


def finalize_simulation_result(
    *,
    mechanism: object,
    result: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
    temperature_schedule: TemperatureScheduleProtocol | Mapping[str, object] | None,
    simulation_plan: SimulationPlan | Mapping[str, Any] | None,
    parameters: tuple[tuple[str, float, str], ...] = (),
    preparation_warnings: list[str] | tuple[str, ...] = (),
) -> SimulationResultValue:
    """Admit one solver output through canonical row finalization.

    The solver's direct identity arrays and actual executed-segment catalog are
    transported unchanged into :class:`SimulationResultValue`; this boundary
    never repairs or reconstructs row identity.
    """

    (
        segment_ids,
        segment_keys,
        discontinuity_ids,
    ) = _required_direct_row_identity(result)
    result_time = np.asarray(getattr(result, "t"), dtype=float)
    if result_time.ndim != 1 or result_time.size == 0:
        raise ValueError("solver result time must be a nonempty 1D array.")
    if np.any(~np.isfinite(result_time)) or np.any(
        np.diff(result_time) <= 0.0
    ):
        raise ValueError("solver result time must be finite and strictly increasing.")
    if (
        segment_ids.ndim != 1
        or segment_ids.size != result_time.size
        or len(segment_keys) != result_time.size
        or discontinuity_ids.ndim != 1
        or discontinuity_ids.size != result_time.size
    ):
        raise ValueError(
            "solver result identity must align with unique result rows."
        )
    finalized = evaluate_simulation_result_algebra(
        mechanism=mechanism,
        result=result,
        species_names=species_names,
        initials_for_algebra=initials_for_algebra,
        temperature_schedule=temperature_schedule,
        simulation_plan=simulation_plan,
    )
    solver_provenance = getattr(result, "provenance", None)
    if not isinstance(solver_provenance, Mapping):
        raise TypeError("solver result provenance must be a mapping.")
    return SimulationResultValue(
        t=result_time,
        series={
            str(name): finalized.y[index, :]
            for index, name in enumerate(finalized.species_names)
        },
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        discontinuity_ids=discontinuity_ids,
        scalars=finalized.algebra_scalars,
        parameters=parameters,
        provenance=SimulationResultProvenance(
            solver=dict(solver_provenance),
            preparation_warnings=tuple(preparation_warnings),
            algebra_errors=tuple(finalized.algebra_errors),
            warnings=tuple(finalized.warnings),
        ),
        intervention_execution=getattr(result, "intervention_execution"),
    )


def build_finalized_simulation_result_payload(
    *,
    mechanism: object,
    result: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
    temperature_schedule: TemperatureScheduleProtocol | Mapping[str, object] | None,
    simulation_plan: SimulationPlan | Mapping[str, Any] | None,
    preparation_warnings: list[str] | tuple[str, ...],
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    include_mechanism: bool,
    parameters: tuple[tuple[str, float, str], ...] = (),
) -> dict[str, Any]:
    admitted = finalize_simulation_result(
        mechanism=mechanism,
        result=result,
        species_names=species_names,
        initials_for_algebra=initials_for_algebra,
        temperature_schedule=temperature_schedule,
        simulation_plan=simulation_plan,
        parameters=parameters,
        preparation_warnings=preparation_warnings,
    )
    warnings = [
        build_simulation_failure(
            "preparation_warning",
            str(message),
            details={"stage": "prepare_run_context"},
        )
        for message in list(preparation_warnings or [])
    ]
    warnings.extend(dict(value) for value in admitted.provenance.warnings)

    builder = build_simulation_success_payload if include_mechanism else build_secondary_simulation_success_payload
    kwargs: dict[str, Any] = {
        "result": admitted,
        "base_species_count": len(species_names),
        "warnings": warnings,
        "solver": str(solver),
        "mechanism_text": str(mechanism_text or ""),
        "solver_config": dict(solver_config or {}),
    }
    if include_mechanism:
        kwargs["mechanism"] = mechanism
    return builder(**kwargs)
