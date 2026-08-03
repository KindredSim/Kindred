"""Exact immutable input values for DSL mechanism construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

from kindred.core.intervention_schedule import InterventionSchedule

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
    arrhenius_A: float | None
    arrhenius_Ea_J_per_mol: float | None
    eyring_dG_act_J_per_mol: float | None
    kappa: float | None
    standard_conc_M: float | None
    dG_eq_J_per_mol: float | None
    Keq_input: float | None
    explicit_rates: tuple[float, ...]
    user_kf_explicit: bool
    user_kr_explicit: bool
    source_energy_unit: str | None

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
        for field in (
            "reversible",
            "is_equilibrium",
            "user_kf_explicit",
            "user_kr_explicit",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be an exact bool.")


@dataclass(frozen=True, slots=True)
class DslMechanismBuildInput:
    """The sole nominal value accepted by the mechanism builder."""

    version: int
    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    kappa_global: float
    temperature_schedule_json: str | None
    intervention_schedule: InterventionSchedule | None
    steps: tuple[DslBuildStep, ...]
    algebra_lines: tuple[str, ...]
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
                self.kappa_global,
            )
        ) or (
            not math.isfinite(self.temperature_K)
            or self.temperature_K <= 0.0
            or not math.isfinite(self.standard_conc_M)
            or self.standard_conc_M <= 0.0
            or not math.isfinite(self.kappa_global)
            or self.kappa_global <= 0.0
        ):
            raise ValueError(
                "DSL mechanism build temperature, standard concentration, and "
                "kappa must be positive and finite."
            )
        if (
            self.temperature_schedule_json is not None
            and type(self.temperature_schedule_json) is not str
        ):
            raise TypeError(
                "DSL mechanism build temperature schedule must be canonical JSON."
            )
        if (
            self.intervention_schedule is not None
            and type(self.intervention_schedule) is not InterventionSchedule
        ):
            raise TypeError(
                "DSL mechanism build intervention schedule must be exact "
                "InterventionSchedule."
            )
        steps = _require_exact_tuple(self.steps, "steps")
        if any(type(entry) is not DslBuildStep for entry in steps):
            raise TypeError("steps must contain exact DslBuildStep values.")
        algebra_lines = _require_exact_tuple(
            self.algebra_lines,
            "algebra_lines",
        )
        if any(type(line) is not str for line in algebra_lines):
            raise TypeError("algebra_lines must contain exact strings.")
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
