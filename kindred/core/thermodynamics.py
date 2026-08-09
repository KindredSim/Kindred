"""Neutral thermodynamic primitives shared by core scientific policies."""

from __future__ import annotations

import math

from .constants import R

__all__ = ["K_from_deltaG_eq"]


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
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("T must be positive and finite")
    if isinstance(dG_eq_J_per_mol, bool):
        raise ValueError("dG_eq_J_per_mol must be numeric")
    try:
        delta_g = float(dG_eq_J_per_mol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dG_eq_J_per_mol must be numeric") from exc
    if not math.isfinite(delta_g):
        raise ValueError("dG_eq_J_per_mol must be finite")
    try:
        value = math.exp(-delta_g / (R * temperature))
    except OverflowError as exc:
        raise ValueError(
            "equilibrium constant overflows the finite real domain"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("equilibrium constant must be positive and finite")
    return float(value)
