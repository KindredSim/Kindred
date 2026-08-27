"""Immutable input values for DSL mechanism construction."""

from __future__ import annotations

from dataclasses import dataclass

from ..temperature import TemperatureSchedule

__all__ = [
    "DslBuildStep",
    "DslMechanismBuildInput",
]


@dataclass(frozen=True, slots=True)
class DslBuildStep:
    reactants: tuple[tuple[str, float], ...]
    products: tuple[tuple[str, float], ...]
    reversible: bool
    kf: float
    kr: float | None
    model: str
    is_equilibrium: bool
    arrhenius_A: float | None = None
    arrhenius_Ea_J_per_mol: float | None = None
    eyring_dG_act_J_per_mol: float | None = None
    kappa: float = 1.0
    standard_conc_M: float | None = None
    dG_eq_J_per_mol: float | None = None
    Keq: float | None = None
    Keq_input: float | None = None
    user_kf_explicit: bool = False
    user_kr_explicit: bool = False
    degeneracy: int = 1

@dataclass(frozen=True, slots=True)
class DslMechanismBuildInput:
    """The sole nominal value accepted by the mechanism builder."""

    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    temperature_schedule: TemperatureSchedule | None
    steps: tuple[DslBuildStep, ...]
    initials: tuple[tuple[str, float], ...]
