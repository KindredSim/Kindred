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
    "source_coordinate_name_pattern",
    "source_coordinate_name_parts",
]


@dataclass(frozen=True, slots=True)
class DslSourceCoordinateCapabilities:
    """The one role catalogue shared by every coordinate projection."""

    role: str
    continuous: bool
    algebra_target: bool
    correction_target: bool
    grouped_model_target: bool = False


# This is intentionally the only role-capability table.  A role may be
# present in the admitted source map without being an independent runtime
# coordinate (for example explicit Eyring degeneracy).
_SOURCE_COORDINATE_CAPABILITIES: Mapping[str, DslSourceCoordinateCapabilities] = MappingProxyType(
    {
        role: DslSourceCoordinateCapabilities(
            role=role,
            continuous=True,
            algebra_target=True,
            correction_target=role in {"kf", "kr", "Keq", "dG_eq"},
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
            continuous=False,
            algebra_target=False,
            correction_target=False,
            grouped_model_target=False,
        ),
        # ``param`` is the right-hand side of a ``param`` line, the source of an
        # independent value (a scalar or a reaction-parameter input).  It is
        # not a chemistry coordinate and therefore does not participate in
        # symbolic, mechanism, or correction projections.
        "param": DslSourceCoordinateCapabilities(
            role="param",
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


def source_coordinate_name_pattern() -> re.Pattern[str]:
    """Return the canonical indexed-name pattern used by consumers."""

    return _SOURCE_COORDINATE_NAME_RE


def source_coordinate_capabilities(role: str) -> DslSourceCoordinateCapabilities:
    """Resolve one admitted role's immutable capability record."""

    return _SOURCE_COORDINATE_CAPABILITIES[role]


def canonical_source_coordinate_name(step_index: int, role: str) -> str:
    """Return the canonical local identity for one authored source role.

    Callers pass a parser step index (from 1) and an indexed chemistry role.
    """

    return f"{role}{step_index}"


def source_coordinate_name_parts(name: str) -> tuple[str, int] | None:
    """Parse a canonical source identity without inferring an owner."""

    # The pattern admits only canonical spellings (no leading zeros).
    match = _SOURCE_COORDINATE_NAME_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("role"), int(match.group("step"))


@dataclass(frozen=True, slots=True)
class DslSourceCoordinate:
    """One immutable value span admitted from exact DSL source bytes.

    The parser builds it from a canonical role/index name (or a ``param``
    name) and the nonempty value span of its own source line.
    """

    canonical_name: str
    role: str
    physical_line_index: int
    value_start: int
    value_end: int

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
    standard_conc_M: float
    dG_eq_J_per_mol: float | None = None
    Keq_input: float | None = None
    source_coordinates: tuple["DslSourceCoordinate", ...] = ()
    # The authored reaction line, which a refusal of its calculated rates shows.
    line: int | None = None
    line_text: str | None = None


@dataclass(frozen=True, slots=True)
class DslMechanismBuildInput:
    """The sole nominal value accepted by the mechanism builder."""

    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    temperature_schedule: TemperatureSchedule | None
    steps: tuple[DslBuildStep, ...]
