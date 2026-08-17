"""Neutral thermodynamic primitives shared by core scientific policies."""

from __future__ import annotations

import math

from .constants import R
from .species import validate_c0, validate_temperature
from .units import admit_energy_j_per_mol, admit_keq

__all__ = ["K_from_deltaG_eq", "concentration_Keq_from_deltaG_eq"]


def K_from_deltaG_eq(dG_eq_J_per_mol: float, T: float) -> float:
    """Return the equilibrium constant from standard Gibbs energy.

    The primitive deliberately depends only on physical constants and scalar
    validation so kinetics and equilibrium-authority modules can both consume
    it without importing one another.
    """

    if isinstance(T, bool):
        raise ValueError("T must be numeric")
    try:
        temperature = float(T)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("T must be numeric") from exc
    temperature = validate_temperature(temperature, field="T")
    if isinstance(dG_eq_J_per_mol, bool):
        raise ValueError("dG_eq_J_per_mol must be numeric")
    try:
        delta_g = float(dG_eq_J_per_mol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dG_eq_J_per_mol must be numeric") from exc
    delta_g = admit_energy_j_per_mol(delta_g, field="dG_eq_J_per_mol")
    try:
        value = math.exp(-delta_g / (R * temperature))
    except OverflowError as exc:
        raise ValueError(
            "equilibrium constant overflows the finite real domain"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("equilibrium constant must be positive and finite")
    return float(value)


def concentration_Keq_from_deltaG_eq(
    dG_eq_J_per_mol: float,
    T: float,
    standard_conc_M: float,
    delta_n: float,
) -> float:
    """Lower standard-state ``dG_eq`` into concentration-form ``Keq``."""
    standard = validate_c0(standard_conc_M, field="C0")
    try:
        exponent = float(delta_n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("delta_n must be finite") from exc
    if not math.isfinite(exponent):
        raise ValueError("delta_n must be finite")
    standard_constant = K_from_deltaG_eq(dG_eq_J_per_mol, T)
    try:
        concentration_constant = standard_constant * math.pow(standard, exponent)
    except (OverflowError, ValueError) as exc:
        raise ValueError("concentration-form equilibrium constant is not finite") from exc
    return admit_keq(concentration_constant, field="concentration-form Keq")
