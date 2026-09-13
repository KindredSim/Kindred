"""Immutable input values for DSL mechanism construction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import re
from typing import Mapping

from ..temperature import TemperatureSchedule
from ..kinetics import ForwardRateModel

__all__ = [
    "DslBuildStep",
    "DslMechanismBuildInput",
    "DslSourceCoordinate",
    "DslSourceCoordinateCapabilities",
    "canonical_source_coordinate_name",
    "source_coordinate_capabilities",
    "source_coordinate_roles",
    "source_coordinate_name_pattern",
    "source_coordinate_name_parts",
]


@dataclass(frozen=True, slots=True)
class DslSourceCoordinateCapabilities:
    """The one role catalogue shared by every coordinate projection."""

    role: str
    exact_symbol: bool
    continuous: bool
    algebra_target: bool
    correction_target: bool
    equilibrium_authority: bool = False
    grouped_model_target: bool = False


# This is intentionally the only role-capability table.  A role may be
# present in the admitted source map without being an independent runtime
# coordinate (for example explicit Eyring degeneracy).
_SOURCE_COORDINATE_CAPABILITIES: Mapping[str, DslSourceCoordinateCapabilities] = MappingProxyType(
    {
        role: DslSourceCoordinateCapabilities(
            role=role,
            exact_symbol=True,
            continuous=True,
            algebra_target=True,
            correction_target=True,
            equilibrium_authority=role in {"kf", "kr", "Keq", "dG_eq"},
            grouped_model_target=role in {"A", "Ea", "dG_act", "dG_eq", "kappa"},
        )
        for role in (
            "k",
            "kf",
            "kr",
            "Keq",
            "A",
            "Ea",
            "dG_act",
            "dG_eq",
            "kappa",
        )
    }
    | {
        "degeneracy": DslSourceCoordinateCapabilities(
            role="degeneracy",
            exact_symbol=True,
            continuous=False,
            algebra_target=False,
            correction_target=False,
            grouped_model_target=False,
        ),
        # ``param`` is source provenance for an independent scalar Algebra
        # declaration.  It is not a chemistry coordinate and therefore does
        # not participate in symbolic, mechanism, or correction projections.
        "param": DslSourceCoordinateCapabilities(
            role="param",
            exact_symbol=False,
            continuous=True,
            algebra_target=False,
            correction_target=False,
            grouped_model_target=False,
        ),
    }
)

_SOURCE_COORDINATE_NAME_RE = re.compile(
    r"^(?P<role>degeneracy|dG_act|dG_eq|kappa|Keq|kf|kr|Ea|A|k)(?P<step>[1-9]\d*)$"
)


def source_coordinate_roles() -> frozenset[str]:
    """Return all admitted source roles from the central immutable catalogue."""

    return frozenset(_SOURCE_COORDINATE_CAPABILITIES)


def source_coordinate_name_pattern() -> re.Pattern[str]:
    """Return the canonical indexed-name pattern used by consumers."""

    return _SOURCE_COORDINATE_NAME_RE


def source_coordinate_capabilities(role: str) -> DslSourceCoordinateCapabilities:
    """Resolve one admitted role's immutable capability record."""

    try:
        return _SOURCE_COORDINATE_CAPABILITIES[str(role)]
    except KeyError as exc:
        raise ValueError(f"Source coordinate role is not admitted: {role!r}.") from exc


def canonical_source_coordinate_name(step_index: int, role: str) -> str:
    """Return the canonical local identity for one authored source role."""

    try:
        index = int(step_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Source coordinate step index must be an integer.") from exc
    if index <= 0:
        raise ValueError("Source coordinate step index must be positive.")
    capabilities = source_coordinate_capabilities(role)
    if not capabilities.exact_symbol:
        raise ValueError(
            f"Source coordinate role {capabilities.role!r} has no indexed identity."
        )
    return f"{capabilities.role}{index}"


def source_coordinate_name_parts(name: str) -> tuple[str, int] | None:
    """Parse a canonical source identity without inferring an owner."""

    match = _SOURCE_COORDINATE_NAME_RE.fullmatch(str(name))
    if match is None:
        return None
    role = str(match.group("role"))
    step = int(match.group("step"))
    if canonical_source_coordinate_name(step, role) != str(name):
        return None
    return role, step


@dataclass(frozen=True, slots=True)
class DslSourceCoordinate:
    """One immutable value span admitted from exact DSL source bytes."""

    canonical_name: str
    role: str
    physical_line_index: int
    value_start: int
    value_end: int

    def __post_init__(self) -> None:
        if (
            type(self.canonical_name) is not str
            or not self.canonical_name
            or self.canonical_name != self.canonical_name.strip()
        ):
            raise ValueError("DSL source coordinate name must be nonempty text.")
        if (
            type(self.role) is not str
            or not self.role
            or self.role != self.role.strip()
            or self.role not in source_coordinate_roles()
        ):
            raise ValueError("DSL source coordinate role is not admitted.")
        if self.role != "param":
            parts = source_coordinate_name_parts(self.canonical_name)
            if parts is None or parts[0] != self.role:
                raise ValueError(
                    "DSL source coordinate name must be its canonical role/index identity."
                )
        if type(self.physical_line_index) is not int or self.physical_line_index < 0:
            raise ValueError(
                "DSL source coordinate physical line index must be non-negative."
            )
        if type(self.value_start) is not int or type(self.value_end) is not int:
            raise TypeError("DSL source coordinate offsets must be exact integers.")
        if self.value_start < 0 or self.value_end <= self.value_start:
            raise ValueError(
                "DSL source coordinate value span must be half-open and nonempty."
            )

    @property
    def capabilities(self) -> DslSourceCoordinateCapabilities:
        return source_coordinate_capabilities(self.role)


@dataclass(frozen=True, slots=True)
class DslBuildStep:
    reactants: tuple[tuple[str, float], ...]
    products: tuple[tuple[str, float], ...]
    reversible: bool
    forward_model: ForwardRateModel
    kr: float | None
    is_equilibrium: bool
    standard_conc_M: float
    dG_eq_J_per_mol: float | None = None
    Keq_input: float | None = None
    source_coordinates: tuple["DslSourceCoordinate", ...] = ()


@dataclass(frozen=True, slots=True)
class DslMechanismBuildInput:
    """The sole nominal value accepted by the mechanism builder."""

    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    temperature_schedule: TemperatureSchedule | None
    steps: tuple[DslBuildStep, ...]
    initials: tuple[tuple[str, float], ...]
