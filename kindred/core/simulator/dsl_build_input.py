"""Exact immutable input values for DSL mechanism construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..temperature import TemperatureSchedule

__all__ = [
    "DslBuildStep",
    "DslMechanismBuildInput",
]


def _require_exact_tuple(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple.")
    return value


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
    Keq_input: float | None = None
    explicit_rates: tuple[float, ...] = ()
    user_kf_explicit: bool = False
    user_kr_explicit: bool = False
    degeneracy: int = 1

    def __post_init__(self) -> None:
        for field in ("reactants", "products"):
            entries = _require_exact_tuple(getattr(self, field), field)
            for entry in entries:
                if (
                    type(entry) is not tuple
                    or len(entry) != 2
                    or type(entry[0]) is not str
                    or type(entry[1]) is not float
                ):
                    raise TypeError(
                        f"{field} must contain exact (str, float) tuples."
                    )
        rates = _require_exact_tuple(self.explicit_rates, "explicit_rates")
        if any(type(rate) is not float for rate in rates):
            raise TypeError("explicit_rates must contain exact floats.")
        if type(self.kf) is not float or not math.isfinite(self.kf) or self.kf <= 0.0:
            raise ValueError("kf must be positive and finite.")
        for field in (
            "kr",
            "arrhenius_A",
            "arrhenius_Ea_J_per_mol",
            "eyring_dG_act_J_per_mol",
            "standard_conc_M",
            "dG_eq_J_per_mol",
            "Keq_input",
        ):
            value = getattr(self, field)
            if value is not None and (
                type(value) is not float or not math.isfinite(value)
            ):
                raise TypeError(f"{field} must be an exact finite float or None.")
        if type(self.model) is not str or self.model not in {
            "Constant",
            "Arrhenius",
            "Eyring",
        }:
            raise ValueError("model must be Constant, Arrhenius, or Eyring.")
        for field in (
            "reversible",
            "is_equilibrium",
            "user_kf_explicit",
            "user_kr_explicit",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be an exact bool.")
        if type(self.degeneracy) is not int:
            raise TypeError("degeneracy must be an exact int.")
        if self.degeneracy <= 0:
            raise ValueError("degeneracy must be a positive integer.")
        if (
            type(self.kappa) is not float
            or not math.isfinite(self.kappa)
            or self.kappa <= 0.0
        ):
            raise ValueError("kappa must be positive and finite.")


@dataclass(frozen=True, slots=True)
class DslMechanismBuildInput:
    """The sole nominal value accepted by the mechanism builder."""

    version: int
    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    temperature_schedule: TemperatureSchedule | None
    steps: tuple[DslBuildStep, ...]
    initials: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TypeError("DSL mechanism build input version must be exact int 1.")
        if type(self.energy_unit) is not str:
            raise TypeError("DSL mechanism build energy_unit must be an exact str.")
        if any(
            type(value) is not float
            for value in (
                self.temperature_K,
                self.standard_conc_M,
            )
        ) or (
            not math.isfinite(self.temperature_K)
            or self.temperature_K <= 0.0
            or not math.isfinite(self.standard_conc_M)
            or self.standard_conc_M <= 0.0
        ):
            raise ValueError(
                "DSL mechanism build temperature and standard concentration "
                "must be positive and finite."
            )
        if self.temperature_schedule is not None and type(self.temperature_schedule) is not TemperatureSchedule:
            raise TypeError(
                "DSL mechanism build temperature schedule must be exact TemperatureSchedule or None."
            )
        steps = _require_exact_tuple(self.steps, "steps")
        if any(type(entry) is not DslBuildStep for entry in steps):
            raise TypeError("steps must contain exact DslBuildStep values.")
        initials = _require_exact_tuple(self.initials, "initials")
        for entry in initials:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or type(entry[1]) is not float
            ):
                raise TypeError(
                    "initials must contain exact (str, float) tuples."
                )
