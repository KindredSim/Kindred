"""
Units handling for Kindred.

This module defines the canonical unit conventions shared across Kindred.

Contracts
---------
- Canonical internal energy unit: joules per mol (J/mol).
- External selectable units: "kcal/mol" or "kJ/mol".
- Temperature in kelvin (K), default 298.15.
- Standard state concentration default: 1.0 M.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from fractions import Fraction

from .quantity_units import parse_unit, convert_value, scientific_unit, conversion_scale

# Category-specific admission limits.  These are deliberately explicit rather
# than one universal scalar envelope: concentrations, rates, energies, and
# standard-state values have different physical meanings and units.
RATE_MIN = 1.0e-20
RATE_MAX = 1.0e20
ARRHENIUS_A_MAX = 1.0e40
ENERGY_ABS_MAX_J_PER_MOL = 1.0e20
KEQ_MIN = 1.0e-20
KEQ_MAX = 1.0e20
ORDINARY_PARAMETER_MIN_ABS = 1.0e-30
ORDINARY_PARAMETER_MAX_ABS = 1.0e30


class OrdinaryParameterDomainError(ValueError):
    """A finite independent scalar lies outside its scientific input domain."""

    def __init__(self, field: str):
        self.field = field
        super().__init__(
            f"{field} must be exact zero or have an absolute value between "
            f"{ORDINARY_PARAMETER_MIN_ABS:g} and {ORDINARY_PARAMETER_MAX_ABS:g} "
            "in scientific units."
        )


def admit_ordinary_parameter(value: object, *, field: str = "parameter") -> float:
    """Admit an independent ordinary parameter already in scientific units."""
    if type(value) is bool:
        raise ValueError(f"{field} must be finite and numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number != 0.0 and not ORDINARY_PARAMETER_MIN_ABS <= abs(number) <= ORDINARY_PARAMETER_MAX_ABS:
        raise OrdinaryParameterDomainError(field)
    return number


def ordinary_parameter_input_bounds(unit: str) -> tuple[float, float]:
    """Positive closed input component, projected inward into editable units."""
    editable = parse_unit(unit)
    scientific = scientific_unit(editable.dimensions)
    conversion_scale(editable, scientific).check_exact_work()
    low = convert_value(ORDINARY_PARAMETER_MIN_ABS, scientific, editable)
    high = convert_value(ORDINARY_PARAMETER_MAX_ABS, scientific, editable)
    while convert_value(low, editable, scientific) < ORDINARY_PARAMETER_MIN_ABS:
        low = math.nextafter(low, math.inf)
    while convert_value(high, editable, scientific) > ORDINARY_PARAMETER_MAX_ABS:
        high = math.nextafter(high, 0.0)
    return low, high

def _admit_finite_range(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
    allow_zero: bool = False,
) -> float:
    if type(value) is bool:
        raise ValueError(f"{field} must be finite and numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number == 0.0 and allow_zero:
        return 0.0
    if number < minimum or number > maximum:
        raise ValueError(
            f"{field} must be between {minimum:g} and {maximum:g}"
        )
    return float(number)


def admit_rate(
    value: object,
    *,
    field: str = "rate",
    allow_zero: bool = False,
) -> float:
    return _admit_finite_range(
        value,
        field=field,
        minimum=RATE_MIN,
        maximum=RATE_MAX,
        allow_zero=allow_zero,
    )


def admit_arrhenius_prefactor(value: object, *, field: str = "A") -> float:
    return _admit_finite_range(
        value,
        field=field,
        minimum=RATE_MIN,
        maximum=ARRHENIUS_A_MAX,
    )


def admit_energy_j_per_mol(value: object, *, field: str = "energy") -> float:
    if type(value) is bool:
        raise ValueError(f"{field} must be finite and numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and numeric") from exc
    if not math.isfinite(number) or abs(number) > ENERGY_ABS_MAX_J_PER_MOL:
        raise ValueError(
            f"{field} absolute value must be at most {ENERGY_ABS_MAX_J_PER_MOL:g} J/mol"
        )
    return float(number)


def admit_keq(value: object, *, field: str = "Keq") -> float:
    return _admit_finite_range(
        value,
        field=field,
        minimum=KEQ_MIN,
        maximum=KEQ_MAX,
    )


def equilibrium_constant_unit(delta_n: object, *, conc_unit: str = "M") -> str:
    """Return the concentration-form unit ``M^delta_n`` for public Keq."""
    exponent = Fraction(str(delta_n))
    return parse_unit(conc_unit).power(exponent).symbol


def rate_constant_unit(
    order: object,
    *,
    conc_unit: str = "M",
    time_unit: str = "s",
) -> str:
    """Return ``M^(1-order)/s`` for one directional kinetic order."""
    exponent = Fraction(str(order))
    if exponent < 0:
        raise ValueError("reaction order must be nonnegative")
    return parse_unit(conc_unit).power(1 - exponent).divide(parse_unit(time_unit)).symbol


# ---------- Units model ----------

@dataclass(frozen=True)
class UnitsModel:
    """
    Portable container for numeric unit preferences.

    Only energy units are actively converted here; temperature/standard
    state values are carried through for provenance.
    """
    energy_unit: str = "kJ/mol"           # or "kcal/mol"
    temperature_K: float = 298.15
    standard_conc_M: float = 1.0

    def __post_init__(self) -> None:
        from .species import validate_c0, validate_temperature

        object.__setattr__(
            self,
            "temperature_K",
            validate_temperature(self.temperature_K, field="temperature_K"),
        )
        standard = validate_c0(self.standard_conc_M, field="C0")
        object.__setattr__(self, "standard_conc_M", standard)

    def from_jmol(self, value: float) -> float:
        """Convert from canonical J/mol to current energy_unit."""
        return convert_value(value, parse_unit("J/mol"), parse_unit(self.energy_unit))
