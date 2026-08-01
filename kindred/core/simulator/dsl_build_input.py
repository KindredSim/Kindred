"""Exact immutable input values for DSL mechanism construction."""

from __future__ import annotations

from dataclasses import dataclass

from kindred.core.intervention_schedule import InterventionSchedule

__all__ = [
    "DslBuildEdge",
    "DslBuildState",
    "DslBuildStep",
    "DslMechanismBuildInput",
]


def _require_exact_tuple(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple.")
    return value


@dataclass(frozen=True, slots=True)
class DslBuildState:
    name: str
    kind: str
    energy_jmol: float
    degeneracy: float
    standard_state: str
    members: tuple[str, ...] | None
    std_conc_product_M: float | None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("DSL build state name must be a nonempty exact str.")
        if type(self.kind) is not str:
            raise TypeError("DSL build state kind must be an exact str.")
        if self.members is not None:
            members = _require_exact_tuple(self.members, "DSL build state members")
            if any(type(member) is not str for member in members):
                raise TypeError(
                    "DSL build state members must contain exact strings."
                )


@dataclass(frozen=True, slots=True)
class DslBuildEdge:
    state_a: str
    state_b: str

    def __post_init__(self) -> None:
        if type(self.state_a) is not str or type(self.state_b) is not str:
            raise TypeError("DSL build edge endpoints must be exact strings.")


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
    cm_id: str | None
    cm_std_ratio: float | None
    generated_computational_mode: bool
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
            "generated_computational_mode",
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
    states: tuple[DslBuildState, ...]
    edges: tuple[DslBuildEdge, ...]
    steps: tuple[DslBuildStep, ...]
    algebra_lines: tuple[str, ...]
    initials: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TypeError("DSL mechanism build input version must be exact int 1.")
        if type(self.energy_unit) is not str:
            raise TypeError("DSL mechanism build energy_unit must be an exact str.")
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
        for field, item_type in (
            ("states", DslBuildState),
            ("edges", DslBuildEdge),
            ("steps", DslBuildStep),
        ):
            entries = _require_exact_tuple(getattr(self, field), field)
            if any(type(entry) is not item_type for entry in entries):
                raise TypeError(
                    f"{field} must contain exact {item_type.__name__} values."
                )
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
