"""Typed GUI coordinates for mechanism parameters.

The GUI never scans DSL text to invent parameter ownership.  This module
parses the accepted source once, projects the core's typed rate families, and
returns the one coordinate representation consumed by Slider and Fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_coordinates import admit_physical_coordinate_value
from kindred.core.quantity_units import parse_unit, scientific_unit, convert_value
from kindred.core.simulation_parameters import enumerate_simulation_parameters
from kindred.core.units import (
    ARRHENIUS_A_MAX,
    ENERGY_ABS_MAX_J_PER_MOL,
    KEQ_MIN,
    KEQ_MAX,
    RATE_MIN,
    RATE_MAX,
    admit_keq,
    admit_rate,
    admit_ordinary_parameter,
    ordinary_parameter_input_bounds,
    UnitsModel,
)

__all__ = [
    "GuiParameterCoordinate",
    "GuiParameterScientificCategory",
    "admit_gui_parameter_value",
    "enumerate_parameter_coordinates",
]


GuiParameterScientificCategory = Literal[
    "rate",
    "equilibrium_constant",
    "finite_scalar",
]


def admit_gui_parameter_value(
    value: object,
    *,
    scientific_category: GuiParameterScientificCategory,
    allow_exact_zero: bool,
    role: str,
    unit: str,
    field: str,
) -> float:
    """Admit one GUI/Fit scalar through its accepted scientific authority."""

    if type(allow_exact_zero) is not bool:
        raise TypeError("allow_exact_zero must be a bool.")
    if scientific_category == "rate":
        return admit_rate(
            value,
            field=field,
            allow_zero=allow_exact_zero,
        )
    if scientific_category == "equilibrium_constant":
        if allow_exact_zero:
            raise ValueError("Equilibrium constants cannot permit exact zero.")
        return admit_keq(value, field=field)
    if scientific_category == "finite_scalar":
        if type(value) is bool or isinstance(value, str):
            raise TypeError(f"{field} must be numeric.")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{field} must be numeric.") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{field} must be finite.")
        input_unit = parse_unit(unit)
        scientific = convert_value(parsed, input_unit, scientific_unit(input_unit.dimensions))
        if role != "scalar":
            admit_physical_coordinate_value(
                scientific, role=role, field=field,
            )
            # GUI values retain authored units; only scientific execution
            # consumes the admission function's J/mol energy result.
            return parsed
        if parsed == 0.0 and not allow_exact_zero:
            raise ValueError(f"{field} must be nonzero.")
        admit_ordinary_parameter(scientific, field=field)
        return float(parsed)
    raise ValueError(f"Unknown GUI parameter scientific category: {scientific_category!r}.")


def _positive_bounds(value: float, *, minimum: float = RATE_MIN, maximum: float = RATE_MAX) -> tuple[float, float]:
    if value > 0.0:
        low = max(value * 0.01, minimum)
        high = min(value * 100.0, maximum)
        if high > low:
            return float(low), float(high)
    return 0.0, max(1.0, abs(value) * 10.0)


def _kappa_bounds(value: float) -> tuple[float, float]:
    """Return a finite positive interval around an admitted Eyring kappa."""

    low = value * 0.01
    if not math.isfinite(low) or low <= 0.0:
        low = math.nextafter(value, 0.0)
        if not math.isfinite(low) or low <= 0.0:
            low = value

    high = value * 100.0
    if not math.isfinite(high):
        high = math.nextafter(value, math.inf)
        if not math.isfinite(high):
            high = value
    return float(low), float(high)


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
    log10_allowed: bool
    family: str
    role: str
    scientific_category: GuiParameterScientificCategory
    allow_exact_zero: bool
    unit: str = "1"

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("GUI parameter name must be nonblank.")
        if type(self.allow_exact_zero) is not bool:
            raise TypeError("GUI parameter exact-zero authority must be a bool.")
        value = admit_gui_parameter_value(
            self.value,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            role=self.role, unit=self.unit,
            field=f"{name} value",
        )
        minimum = admit_gui_parameter_value(
            self.minimum,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            role=self.role, unit=self.unit,
            field=f"{name} minimum",
        )
        maximum = admit_gui_parameter_value(
            self.maximum,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            role=self.role, unit=self.unit,
            field=f"{name} maximum",
        )
        if minimum >= maximum or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} requires minimum <= value <= maximum."
            )
        if self.log10_allowed and (minimum <= 0.0 or value <= 0.0):
            raise ValueError(f"{name} cannot offer Log10 outside a positive domain.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


def enumerate_parameter_coordinates(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[GuiParameterCoordinate, ...]:
    """Add GUI bounds and labels to the core's admitted parameter facts."""
    coordinates: list[GuiParameterCoordinate] = []
    for item in enumerate_simulation_parameters(prepared_dsl):
        value = item.value
        if item.scientific_category in {"rate", "equilibrium_constant"}:
            minimum, maximum = (
                (KEQ_MIN, KEQ_MAX) if item.role == "Keq" else (RATE_MIN, RATE_MAX)
            )
            low, high = (
                _positive_bounds(value, minimum=minimum, maximum=maximum)
                if value >= 0.0 else _signed_bounds(value)
            )
        elif item.role == "A":
            low, high = _positive_bounds(value, minimum=RATE_MIN, maximum=ARRHENIUS_A_MAX)
        elif item.role == "kappa":
            low, high = _kappa_bounds(value)
        elif item.role in {"Ea", "dG_act", "dG_eq"}:
            limit = UnitsModel(energy_unit=item.unit).from_jmol(ENERGY_ABS_MAX_J_PER_MOL)
            low, high = _signed_bounds(value)
            low, high = max(low, -limit), min(high, limit)
        else:
            low, high = _signed_bounds(value)
            floor, limit = ordinary_parameter_input_bounds(item.unit)
            low, high = max(low, -limit), min(high, limit)
            if 0.0 < abs(low) < floor:
                low = 0.0
            if 0.0 < abs(high) < floor:
                high = 0.0
            if low == high:
                low, high = -floor, floor
        display_name = item.name
        if item.context:
            display_name = f"{display_name} — {item.context}"
        coordinates.append(GuiParameterCoordinate(
            name=item.name, display_name=display_name, value=value,
            minimum=low, maximum=high,
            # Rate constants and prefactors span decades; energies and plain
            # scalars are linear quantities.
            log10_allowed=bool(
                (
                    item.scientific_category in {"rate", "equilibrium_constant"}
                    or item.role in {"A", "kappa"}
                )
                and value > 0.0
                and low > 0.0
            ),
            family=item.family, role=item.role,
            scientific_category=item.scientific_category,
            allow_exact_zero=item.allow_exact_zero, unit=item.unit,
        ))
    return tuple(coordinates)
