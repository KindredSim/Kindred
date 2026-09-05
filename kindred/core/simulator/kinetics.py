"""
Kinetics mapping for Kindred: Eyring and Arrhenius, kappa handling, and units.

Current contract
----------------
- Units handling: temperature in K (default 298.15), standard states 1 M or 1 bar.
- Eyring and Arrhenius mapping:
    Eyring unimolecular: k [1/s] = kappa * (k_B*T/h) * exp(-dG_act/(R*T))
    Bimolecular and n-molecular: multiply by 1/(C0^(n-1)) to yield 1/(M^(n−1)*s)
    Arrhenius: k = A * exp(-Ea/(R*T))
    Units inferred from molecularity: 1/(M^(n−1)*s)
- Preview output format: default 3 sig figs, scientific `e±NN`, ROUND_HALF_UP.
- Locked conventions: constants, reaction-local Eyring kappa default 1.0.

Core Arrhenius/Eyring/K(dG_eq) mappings live in :mod:`kindred.core.kinetics`.
This module owns simulator DSL and GUI preview helpers only.
"""

from __future__ import annotations

import math

from ..units import kcalmol_to_jmol, kjmol_to_jmol


__all__ = [
    "normalize_energy_to_J_per_mol",
]


# ------------------------------ energy normalization -------------------------


def normalize_energy_to_J_per_mol(value: float, unit: str | None) -> float:
    """
    Convert energy to canonical J/mol.

    unit ∈ {"J/mol","kJ/mol","kcal/mol"}; if None, assume J/mol.
    """
    if unit is None or unit == "J/mol":
        ej = float(value)
    elif unit == "kJ/mol":
        ej = kjmol_to_jmol(float(value))
    elif unit == "kcal/mol":
        ej = kcalmol_to_jmol(float(value))
    else:
        raise ValueError(f"unsupported energy unit {unit!r}")
    if not math.isfinite(ej):
        raise ValueError("energy must be finite")
    return ej
