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

# Category-specific admission limits.  These are deliberately explicit rather
# than one universal scalar envelope: concentrations, rates, energies, and
# standard-state values have different physical meanings and units.
RATE_MIN = 1.0e-20
RATE_MAX = 1.0e20
ARRHENIUS_A_MAX = 1.0e40
ENERGY_ABS_MAX_J_PER_MOL = 1.0e20
KEQ_MIN = 1.0e-20
KEQ_MAX = 1.0e20

# ---------- scalar converters ----------

def kcalmol_to_jmol(x: float) -> float:
    """Convert kcal/mol -> J/mol."""
    return float(x) * 4184.0

def jmol_to_kcalmol(x: float) -> float:
    """Convert J/mol -> kcal/mol."""
    return float(x) / 4184.0

def kjmol_to_jmol(x: float) -> float:
    """Convert kJ/mol -> J/mol."""
    return float(x) * 1000.0

def jmol_to_kjmol(x: float) -> float:
    """Convert J/mol -> kJ/mol."""
    return float(x) / 1000.0


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
    try:
        exponent = float(delta_n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("equilibrium delta_n must be numeric") from exc
    if not math.isfinite(exponent):
        raise ValueError("equilibrium delta_n must be finite")
    if exponent == 0.0:
        return "1"
    if exponent == 1.0:
        return str(conc_unit)
    if exponent == -1.0:
        return f"1/{conc_unit}"
    if exponent > 0.0:
        return f"{conc_unit}^{format(exponent, '.15g')}"
    return f"1/({conc_unit}^{format(-exponent, '.15g')})"


def rate_constant_unit(
    order: object,
    *,
    conc_unit: str = "M",
    time_unit: str = "s",
) -> str:
    """Return ``M^(1-order)/s`` for one directional kinetic order."""
    try:
        exponent = float(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reaction order must be numeric") from exc
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError("reaction order must be finite and nonnegative")
    power = 1.0 - exponent
    if power == 0.0:
        return f"1/{time_unit}"
    if power == 1.0:
        return f"{conc_unit}/{time_unit}"
    if power == -1.0:
        return f"1/({conc_unit} {time_unit})"
    if power > 0.0:
        return f"{conc_unit}^{format(power, '.15g')}/{time_unit}"
    return f"1/({conc_unit}^{format(-power, '.15g')} {time_unit})"


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
        if self.energy_unit == "kJ/mol":
            return jmol_to_kjmol(value)
        if self.energy_unit == "kcal/mol":
            return jmol_to_kcalmol(value)
        if self.energy_unit == "J/mol":
            return float(value)
        raise ValueError(f"Unsupported energy_unit: {self.energy_unit!r}")
