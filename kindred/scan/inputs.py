"""Consume existing scientific input facts and admission, without owning them."""
from __future__ import annotations

from dataclasses import dataclass
import sys

from kindred.core.simulation_parameters import enumerate_simulation_parameters
from kindred.core.species import (
    CONCENTRATION_MIN_M, CONCENTRATION_MAX_M, TIME_INPUT_MIN_S, TIME_MAX_S,
)
from kindred.scan.config import START, END, ScanConfig, initial_key, input_label, parameter_key
from kindred.scan.intervals import Interval, intersect, normalize


@dataclass(frozen=True)
class ScanInput:
    key: str
    label: str
    value: float
    unit: str
    bounds: tuple[Interval, ...]
    family: str = ""
    role: str = ""


def input_catalog(prepared, config: ScanConfig) -> tuple[ScanInput, ...]:
    positive_concentration = Interval(CONCENTRATION_MIN_M, CONCENTRATION_MAX_M)
    entries = [ScanInput(initial_key(name), input_label(initial_key(name)),
                         config.initials[name], "M", (Interval(0.0, 0.0), positive_concentration))
               for name in prepared.analysis.species_names]
    parameters = enumerate_simulation_parameters(prepared)
    for item in parameters:
        # Bounds are the parameter's scientific domain, already in its input unit.
        floor, limit = item.domain
        positive = Interval(floor, limit, include_lower=floor > 0.0)
        if item.signed and floor == 0.0:
            bounds = (Interval(-limit, limit),)
        elif item.signed:
            bounds = (Interval(-limit, -floor), positive)
        else:
            bounds = (positive,)
        if item.allow_exact_zero and not any(component.contains(0.0) for component in bounds):
            bounds = (Interval(0.0, 0.0),) + bounds
        elif not item.allow_exact_zero:
            bounds = intersect(bounds, (Interval(-sys.float_info.max, 0.0, True, False),
                                        Interval(0.0, sys.float_info.max, False, True)))
        entries.append(ScanInput(parameter_key(item.name), input_label(parameter_key(item.name)),
                                 config.parameters.get(item.name, item.value), item.unit, bounds,
                                 item.family, item.role))
    time_bounds = (Interval(0.0, 0.0), Interval(TIME_INPUT_MIN_S, TIME_MAX_S))
    entries.extend((ScanInput(START, input_label(START), config.t_start, "s", time_bounds),
                    ScanInput(END, input_label(END), config.t_end, "s", time_bounds)))
    return tuple(entries)


def _bounds_text(item: ScanInput) -> str:
    """The scientific bounds of an input, in its unit."""
    parts = (f"{i.lower:g}" if i.lower == i.upper else
             f"{'[' if i.include_lower else '('}{i.lower:g}, {i.upper:g}{']' if i.include_upper else ')'}"
             for i in normalize(item.bounds))
    return " or ".join(parts) + ("" if item.unit == "1" else f" {item.unit}")


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
            raise ValueError(f"{item.label}: allowed ranges do not intersect its scientific bounds, {_bounds_text(item)}.")
        if requested:
            for value in requested.endpoints:
                if not any(value in (i.lower, i.upper) and i.contains(value) for i in intervals):
                    raise ValueError(f"{item.label}: {value:g} is not an included endpoint.")
                if not any(i.contains(value) for i in allowed):
                    raise ValueError(f"{item.label}: selected endpoint {value:g} is outside its scientific bounds, {_bounds_text(item)}.")
        out[item.key] = allowed
    return out
