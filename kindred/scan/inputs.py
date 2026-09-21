"""Consume existing scientific input facts and admission, without owning them."""
from __future__ import annotations

from dataclasses import dataclass
import sys

from kindred.core.simulation_parameters import enumerate_simulation_parameters
from kindred.core.simulator.parameter_coordinates import build_parameter_target_map, _admit_coordinate_value
from kindred.core.species import (
    CONCENTRATION_MIN_M, CONCENTRATION_MAX_M, TIME_MIN_S, TIME_MAX_S,
)
from kindred.core.units import (
    RATE_MIN, RATE_MAX, KEQ_MIN, KEQ_MAX, ARRHENIUS_A_MAX,
    ENERGY_ABS_MAX_J_PER_MOL, UnitsModel, ordinary_parameter_input_bounds,
)
from kindred.scan.config import START, END, ScanConfig, initial_key, parameter_key
from kindred.scan.intervals import Interval, intersect, normalize


@dataclass(frozen=True)
class ScanInput:
    key: str
    label: str
    value: float
    unit: str
    bounds: tuple[Interval, ...]


def input_catalog(prepared, config: ScanConfig) -> tuple[ScanInput, ...]:
    positive_concentration = Interval(CONCENTRATION_MIN_M, CONCENTRATION_MAX_M)
    entries = [ScanInput(initial_key(name), f"Initial {name}",
                         config.initials[name], "M", (Interval(0.0, 0.0), positive_concentration))
               for name in prepared.analysis.species_names]
    parameters = enumerate_simulation_parameters(prepared)
    shell = None
    for item in parameters:
        if item.scientific_category == "rate":
            bounds = (Interval(RATE_MIN, RATE_MAX),)
        elif item.scientific_category == "equilibrium_constant":
            bounds = (Interval(KEQ_MIN, KEQ_MAX),)
        elif item.role == "A":
            bounds = (Interval(RATE_MIN, ARRHENIUS_A_MAX),)
        elif item.role in {"Ea", "dG_act", "dG_eq"}:
            limit = UnitsModel(energy_unit=item.unit).from_jmol(ENERGY_ABS_MAX_J_PER_MOL)
            bounds = (Interval(-limit, limit),)
        elif item.role == "scalar":
            floor, limit = ordinary_parameter_input_bounds(item.unit)
            bounds = (Interval(-limit, -floor), Interval(0.0, 0.0), Interval(floor, limit))
        else:
            bounds = (Interval(-sys.float_info.max, sys.float_info.max),)
        if item.allow_exact_zero and bounds[0].lower > 0:
            bounds = (Interval(0.0, 0.0),) + bounds
        elif not item.allow_exact_zero:
            bounds = intersect(bounds, (Interval(-sys.float_info.max, 0.0, True, False),
                                        Interval(0.0, sys.float_info.max, False, True)))
        if item.role == "kappa":
            if shell is None:
                shell = prepared.mutable_mechanism_shell()
                targets = build_parameter_target_map(shell)
            route, kind, index, role, _ = targets[item.name]
            admitted = []
            # The existing kappa primitive predicate has constant admission on
            # each finite sign component. Consume that predicate; these probes
            # are not a general method for discovering arbitrary input domains.
            for component in bounds:
                try:
                    _admit_coordinate_value(shell, item.name, component.lower / 2 + component.upper / 2,
                                            route=route, kind=kind, index=index, role=role)
                except ValueError:
                    continue
                admitted.append(component)
            bounds = tuple(admitted)
        entries.append(ScanInput(parameter_key(item.name), item.name,
                                 config.parameters.get(item.name, item.value), item.unit, bounds))
    entries.extend((ScanInput(START, "Start time", config.t_start, "s", (Interval(TIME_MIN_S, TIME_MAX_S),)),
                    ScanInput(END, "End time", config.t_end, "s", (Interval(TIME_MIN_S, TIME_MAX_S),))))
    return tuple(entries)


def admitted_ranges(catalog: tuple[ScanInput, ...], config: ScanConfig) -> dict[str, tuple[Interval, ...]]:
    unknown = set(config.ranges) - {item.key for item in catalog}
    if unknown:
        raise ValueError(f"Inputs are no longer independently editable: {', '.join(sorted(unknown))}")
    out = {}
    for item in catalog:
        requested = config.ranges.get(item.key)
        intervals = normalize(requested.intervals) if requested else (Interval(item.value, item.value),)
        allowed = intersect(intervals, item.bounds)
        if not allowed:
            raise ValueError(f"{item.label}: allowed ranges do not intersect its scientific bounds.")
        if requested:
            for value in requested.endpoints:
                if not any(value in (i.lower, i.upper) and i.contains(value) for i in intervals):
                    raise ValueError(f"{item.label}: {value:g} is not an included endpoint.")
                if not any(i.contains(value) for i in allowed):
                    raise ValueError(f"{item.label}: selected endpoint {value:g} is outside scientific bounds.")
        out[item.key] = allowed
    return out
