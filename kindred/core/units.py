"""
Units handling for Kindred.

This module defines the canonical unit conventions shared across Kindred.

Contracts
---------
- Canonical internal energy unit: joules per mol (J/mol).
- External selectable units: "kcal/mol" or "kJ/mol".
- Temperature in kelvin (K), default 298.15.
- Standard state concentration default: 1.0 M.
- Standard state pressure default: 1.0e5 Pa.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    standard_pressure_Pa: float = 1.0e5

    def to_jmol(self, value: float) -> float:
        """Convert from current energy_unit to canonical J/mol."""
        if self.energy_unit == "kJ/mol":
            return kjmol_to_jmol(value)
        if self.energy_unit == "kcal/mol":
            return kcalmol_to_jmol(value)
        if self.energy_unit == "J/mol":
            return float(value)
        raise ValueError(f"Unsupported energy_unit: {self.energy_unit!r}")

    def from_jmol(self, value: float) -> float:
        """Convert from canonical J/mol to current energy_unit."""
        if self.energy_unit == "kJ/mol":
            return jmol_to_kjmol(value)
        if self.energy_unit == "kcal/mol":
            return jmol_to_kcalmol(value)
        if self.energy_unit == "J/mol":
            return float(value)
        raise ValueError(f"Unsupported energy_unit: {self.energy_unit!r}")
