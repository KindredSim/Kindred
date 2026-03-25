"""
Core kinetics formulas shared across Kindred.

This module intentionally contains the fundamental Arrhenius/Eyring and
equilibrium-constant mappings so both the simulator layer and the ODE builder
can depend on a single source of truth without creating cross-layer imports.
"""

from __future__ import annotations

import math

from .constants import R, h, k_B

__all__ = [
    "arrhenius_rate",
    "eyring_prefactor",
    "eyring_rate",
    "K_from_deltaG_eq",
]


def eyring_prefactor(T: float) -> float:
    """(k_B T / h) in 1/s."""
    return (k_B * float(T)) / h


def eyring_rate(
    dG_act_J_per_mol: float,
    T: float,
    *,
    kappa: float = 1.0,
    molecularity: int = 1,
    standard_conc_M: float = 1.0,
) -> float:
    """
    Compute Eyring rate constant.

    - Unimolecular: κ * (k_B T / h) * exp(-ΔG‡/(R T))      [1/s]
    - n-molecular (n >= 2): multiply by 1/(C°^(n-1))        [1/(M^(n-1)*s)]
    """
    T = float(T)
    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("T must be positive and finite")
    if standard_conc_M <= 0.0 or not math.isfinite(standard_conc_M):
        raise ValueError("standard_conc_M must be positive and finite")
    n = int(molecularity)
    if n < 1:
        raise ValueError("molecularity must be >= 1")

    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be positive and finite")

    exponent = -float(dG_act_J_per_mol) / (R * T)
    k = kappa * eyring_prefactor(T) * math.exp(exponent)
    if n >= 2:
        k /= float(standard_conc_M) ** (n - 1)
    return float(k)


def arrhenius_rate(A: float, Ea_J_per_mol: float, T: float) -> float:
    """
    Arrhenius rate constant:

        k = A * exp(-Ea/(R T))

    Notes
    -----
    - A should already carry the correct order unit (1/s or 1/(M^(n-1)*s)).
    - This function does not attempt to infer molecularity from A.
    """
    T = float(T)
    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("T must be positive and finite")
    exponent = -float(Ea_J_per_mol) / (R * T)
    return float(float(A) * math.exp(exponent))


def K_from_deltaG_eq(dG_eq_J_per_mol: float, T: float) -> float:
    """
    Equilibrium constant from standard Gibbs free energy change:

        K = exp(-ΔG° / (R T))
    """
    T = float(T)
    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("T must be positive and finite")
    return float(math.exp(-float(dG_eq_J_per_mol) / (R * T)))

