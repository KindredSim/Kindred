"""Neutral thermodynamic primitives shared by core scientific policies."""

from __future__ import annotations

import math

from .constants import R
from .units import admit_keq

__all__ = ["K_from_deltaG_eq", "concentration_Keq_from_deltaG_eq"]


# dG_eq arrives admitted by DSL parsing or the grouped coordinate owner, the
# temperature by the DSL T directive or its admitted schedule, and the standard
# concentration by DSL parsing; delta_n is the stoichiometric order difference.


def K_from_deltaG_eq(dG_eq_J_per_mol: float, T: float) -> float:
    """Return the equilibrium constant from standard Gibbs energy.

    The primitive deliberately depends only on physical constants so kinetics
    and equilibrium-authority modules can both consume it without importing one
    another.
    """

    try:
        return math.exp(-dG_eq_J_per_mol / (R * T))
    except OverflowError as exc:
        raise ValueError(
            "equilibrium constant overflows the finite real domain"
        ) from exc


def concentration_Keq_from_deltaG_eq(
    dG_eq_J_per_mol: float,
    T: float,
    standard_conc_M: float,
    delta_n: float,
) -> float:
    """Lower standard-state ``dG_eq`` into concentration-form ``Keq``."""
    standard_constant = K_from_deltaG_eq(dG_eq_J_per_mol, T)
    try:
        concentration_constant = standard_constant * math.pow(standard_conc_M, delta_n)
    except (OverflowError, ValueError) as exc:
        raise ValueError("concentration-form equilibrium constant is not finite") from exc
    return admit_keq(concentration_constant, field="The Keq calculated from dG_eq")
