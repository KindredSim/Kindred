"""Typed GUI coordinates for mechanism parameters.

The GUI never scans DSL text to invent parameter ownership.  This module
parses the accepted source once, projects the core's typed rate families, and
returns the one coordinate representation consumed by Slider and Fit; each
coordinate carries its Slider range and its independent default Fit range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.quantity_units import parse_unit, scientific_unit, convert_value
from kindred.core.simulation_parameters import enumerate_simulation_parameters
from kindred.core.validation import six_digit_value
from kindred.core.units import admit_role_value, UnitsModel

__all__ = [
    "GuiParameterCoordinate",
    "admit_gui_parameter_value",
    "enumerate_parameter_coordinates",
]


def admit_gui_parameter_value(
    value: float,
    *,
    allow_exact_zero: bool,
    role: str,
    unit: str,
    field: str,
) -> float:
    """Admit one GUI/Fit value, held in ``unit``, against its role's scientific domain."""

    input_unit = parse_unit(unit)
    scientific = convert_value(value, input_unit, scientific_unit(input_unit.dimensions))
    admit_role_value(scientific, role=role, unit=unit, allow_exact_zero=allow_exact_zero, field=field)
    return value


def _positive_bounds(value: float, *, minimum: float, maximum: float) -> tuple[float, float]:
    if value > 0.0:
        low = max(value * 0.01, minimum)
        high = min(value * 100.0, maximum)
        if high > low:
            return float(low), float(high)
    return 0.0, min(max(1.0, abs(value) * 10.0), maximum)


def _signed_bounds(value: float) -> tuple[float, float]:
    span = max(1.0, abs(value) * 0.5)
    low = value - span
    high = value + span
    if not math.isfinite(low) or low >= value:
        low = value
    if not math.isfinite(high) or high <= value:
        high = value
    if low == high:
        low = math.nextafter(value, -math.inf)
        high = math.nextafter(value, math.inf)
    if not math.isfinite(low):
        low = value
    if not math.isfinite(high):
        high = value
    if not low < high:
        raise ValueError("Finite scalar has no representable finite Fit interval.")
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class GuiParameterCoordinate:
    """One stable GUI/Fit coordinate backed by a typed scientific owner."""

    name: str
    display_name: str
    value: float
    minimum: float
    maximum: float
    fit_minimum: float
    fit_maximum: float
    log10_allowed: bool
    family: str
    role: str
    allow_exact_zero: bool
    unit: str = "1"


def enumerate_parameter_coordinates(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[GuiParameterCoordinate, ...]:
    """Add GUI bounds and labels to the core's admitted parameter facts."""
    coordinates: list[GuiParameterCoordinate] = []
    for item in enumerate_simulation_parameters(prepared_dsl):
        value = item.value
        floor, limit = item.domain
        if not item.signed:
            low, high = _positive_bounds(value, minimum=floor, maximum=limit)
        elif item.role in {"Ea", "dG_act", "dG_eq"}:
            energy = UnitsModel(energy_unit=item.unit)
            if item.role == "dG_eq":
                # Equilibrium energies are signed; the Slider crosses zero.
                half = max(2.0 * abs(value), energy.from_jmol(10_000.0))
                low, high = -half, half
            else:
                # Barriers run from zero to twice the accepted value.
                width = max(2.0 * abs(value), energy.from_jmol(1_000.0))
                low, high = (0.0, width) if value >= 0.0 else (-width, 0.0)
            low, high = max(low, -limit), min(high, limit)
            span = energy.from_jmol(10_000.0)
            fit_low, fit_high = max(value - span, -limit), min(value + span, limit)
        else:
            low, high = _signed_bounds(value)
            low, high = max(low, -limit), min(high, limit)
            if 0.0 < abs(low) < floor:
                low = 0.0
            if 0.0 < abs(high) < floor:
                high = 0.0
            if low == high:
                low, high = -floor, floor
        if item.role not in {"Ea", "dG_act", "dG_eq"}:
            fit_low, fit_high = low, high
        display_name = item.name
        if item.context:
            display_name = f"{display_name} — {item.context}"
        coordinates.append(GuiParameterCoordinate(
            name=item.name, display_name=display_name, value=value,
            minimum=low, maximum=high,
            # Generated Fit bounds are authored at six significant digits.
            fit_minimum=six_digit_value(fit_low), fit_maximum=six_digit_value(fit_high),
            # Positive quantities (rates, Keq, A, kappa) span decades; energies and
            # plain scalars are signed, linear quantities.
            log10_allowed=not item.signed and value > 0.0 and low > 0.0,
            family=item.family, role=item.role,
            allow_exact_zero=item.allow_exact_zero, unit=item.unit,
        ))
    return tuple(coordinates)
