"""Energy-unit normalization for authored DSL energies.

Core Arrhenius/Eyring rate mappings live in :mod:`kindred.core.kinetics`.
"""

from __future__ import annotations

import math

from ..quantity_units import convert_value, parse_unit


__all__ = [
    "normalize_energy_to_J_per_mol",
]


# ------------------------------ energy normalization -------------------------


def normalize_energy_to_J_per_mol(value: float, unit: str | None) -> float:
    """
    Convert energy to canonical J/mol.

    unit ∈ {"J/mol","kJ/mol","kcal/mol"}; if None, assume J/mol.
    """
    ej = convert_value(value, parse_unit(unit or "J/mol"), parse_unit("J/mol"))
    if not math.isfinite(ej):
        raise ValueError("energy must be finite")
    return ej
