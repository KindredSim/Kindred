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

from .quantity_units import parse_unit, convert_value, scientific_unit, conversion_scale

# Category-specific admission limits.  These are deliberately explicit rather
# than one universal scalar envelope: concentrations, rates, energies, and
# standard-state values have different physical meanings and units.
RATE_MAX = 1.0e20
ARRHENIUS_A_MIN = 1.0e-20
ARRHENIUS_A_MAX = 1.0e40
ENERGY_ABS_MAX_J_PER_MOL = 1.0e20
KEQ_MIN = 1.0e-20
KEQ_MAX = 1.0e20
KAPPA_MAX = 1.0e20
ORDINARY_PARAMETER_MIN_ABS = 1.0e-30
ORDINARY_PARAMETER_MAX_ABS = 1.0e30


class OrdinaryParameterDomainError(ValueError):
    """A finite independent scalar lies outside its scientific input domain."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message)


# Scientific magnitude domain and sign per coordinate role; a role not listed is a rate.
_ROLE_DOMAINS = {
    "Keq": (KEQ_MIN, KEQ_MAX, False),
    "A": (ARRHENIUS_A_MIN, ARRHENIUS_A_MAX, False),
    "kappa": (0.0, KAPPA_MAX, False),
    "Ea": (0.0, ENERGY_ABS_MAX_J_PER_MOL, True),
    "dG_act": (0.0, ENERGY_ABS_MAX_J_PER_MOL, True),
    "dG_eq": (0.0, ENERGY_ABS_MAX_J_PER_MOL, True),
    "scalar": (ORDINARY_PARAMETER_MIN_ABS, ORDINARY_PARAMETER_MAX_ABS, True),
}
_RATE_DOMAIN = (0.0, RATE_MAX, False)


def role_domain(role: str) -> tuple[float, float, bool]:
    """Scientific magnitude floor (open when zero), ceiling and sign of one coordinate role."""
    return _ROLE_DOMAINS.get(role, _RATE_DOMAIN)


def admit_role_value(
    value: float, *, role: str, unit: str, allow_exact_zero: bool, field: str,
) -> float:
    """Admit a scientific value of one coordinate; a refusal states its domain in ``unit``."""
    low, high, signed = role_domain(role)
    magnitude = abs(value) if signed else value
    if (value == 0.0 and allow_exact_zero) or (
        (low < magnitude if low == 0.0 else low <= magnitude) and magnitude <= high
    ):
        return value
    floor, limit = input_unit_bounds(unit, low, high)
    label = "" if unit == "1" else f" {unit}"
    if low > 0.0 and signed:
        zero = "be exact zero or " if allow_exact_zero else ""
        message = f"{field} must {zero}have an absolute value between {floor:g} and {limit:g}{label}."
    elif low > 0.0:
        message = f"{field} must be between {floor:g} and {limit:g}{label}."
    elif signed:
        message = f"{field} must have an absolute value of at most {limit:g}{label}."
    else:
        message = f"{field} must be {'zero or positive' if allow_exact_zero else 'positive'} and at most {limit:g}{label}."
    if role == "scalar":
        raise OrdinaryParameterDomainError(field, message)
    raise ValueError(message)


def input_unit_bounds(unit: str, low: float, high: float) -> tuple[float, float]:
    """A nonnegative closed scientific interval, projected inward into an editable unit."""
    editable = parse_unit(unit)
    scientific = scientific_unit(editable.dimensions)
    conversion_scale(editable, scientific).check_exact_work()
    lower = convert_value(low, scientific, editable)
    upper = convert_value(high, scientific, editable)
    while convert_value(lower, editable, scientific) < low:
        lower = math.nextafter(lower, math.inf)
    while convert_value(upper, editable, scientific) > high:
        upper = math.nextafter(upper, 0.0)
    return lower, upper


# Typed and computed values in scientific units, admitted against the role domains above.
def admit_rate(value: float, *, field: str = "rate", allow_zero: bool = False) -> float:
    return admit_role_value(value, role="k", unit="1", allow_exact_zero=allow_zero, field=field)


def admit_arrhenius_prefactor(value: float, *, field: str = "A") -> float:
    return admit_role_value(value, role="A", unit="1", allow_exact_zero=False, field=field)


def admit_energy_j_per_mol(value: float, *, field: str = "energy") -> float:
    # Every energy role shares one domain.
    return admit_role_value(value, role="Ea", unit="J/mol", allow_exact_zero=True, field=field)


def admit_keq(value: float, *, field: str = "Keq") -> float:
    return admit_role_value(value, role="Keq", unit="1", allow_exact_zero=False, field=field)


# ---------- Energy units ----------

_ENERGY_UNITS = {
    "j/mol": "J/mol",
    "kj/mol": "kJ/mol",
    "kcal/mol": "kcal/mol",
}


def normalize_energy_unit(value: str) -> str:
    """Return the canonical spelling of an authored energy unit."""
    canonical = _ENERGY_UNITS.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"unsupported energy_unit {value!r}")
    return canonical


def energy_to_j_per_mol(value: float, unit: str) -> float:
    """Convert an authored energy to the canonical J/mol."""
    return convert_value(value, parse_unit(unit), parse_unit("J/mol"))


# ---------- Units model ----------

@dataclass(frozen=True)
class UnitsModel:
    """
    Portable container for numeric unit preferences.

    Only energy units are actively converted here; temperature/standard
    state values are carried through for provenance.  Its temperature and
    standard concentration arrive admitted with the tab's scientific settings.
    """
    energy_unit: str = "kJ/mol"           # or "kcal/mol"
    temperature_K: float = 298.15
    standard_conc_M: float = 1.0

    def from_jmol(self, value: float) -> float:
        """Convert from canonical J/mol to current energy_unit."""
        return convert_value(value, parse_unit("J/mol"), parse_unit(self.energy_unit))
