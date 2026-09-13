"""
Computational Simulator DSL tools.

Current contract
----------------
- Steps may be bare or prefixed with `reaction:` or `equilibrium:`.
- Irreversible arrows: `->` and `=>`; reversible arrows: `<->` and `<=>`.
- Accepted keys with normalization:
  degeneracy, kappa, C0, dG_act, Ea, dG_eq, k|kf|kr, A,
  energy=kcal/mol|kJ/mol|J/mol
- Packaged preset sources use this same public syntax and remain admitted by
  the same parser.
- Source-level `T=`, `C0=`, and `energy=` override supplied mechanism defaults.
- If dG_eq is provided and Keq is omitted, compute concentration-form
  Keq = exp(-dG_eq/(R*T)) * C0**delta_n with reaction-dependent units.
- Default per step is Eyring; Arrhenius override via A/Ea.
- Units inferred from molecularity (1/(M^(n−1)*s)); bimolecular Eyring divides by C0.

Scope
-----
- Parse a simple line-oriented DSL (no nested syntax).
- Provide immutable analysis and retained prepared execution structures.
- No file access; caller passes text.

Non-goals
---------
- Persisting anything or solver integration.

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import re

from ..equilibrium_rate_authority import (
    validate_equilibrium_rate_authority_flags,
)
from ..mechanism_source import (
    INITIAL_PROVENANCE_PREFIX,
    MechanismAuthoringSource,
    normalize_source_for_analysis,
    source_matches_analysis,
)
from ..kinetics import ArrheniusRateModel, ConstantRateModel, EyringRateModel
from ..units import UnitsModel
from ..units import (
    admit_energy_j_per_mol,
    admit_keq,
    admit_rate,
)
from ..species import validate_c0, validate_concentration, validate_temperature
from ..stoichiometry import directional_order_fraction
from .errors import (
    DSLError as _DSLError,
    invalid_number_error,
    missing_arrow_error,
    invalid_species_term_error,
    empty_stoichiometry_error,
    invalid_keyvalue_pair_error,
)
from .parameter_algebra_spec import classify_parameter_algebra_declaration
from .parameter_namespace import (
    build_namespace_from_mechanism,
)
from .dsl_build_input import (
    DslBuildStep,
    DslSourceCoordinate,
    canonical_source_coordinate_name,
)

if TYPE_CHECKING:
    from ..intervention_schedule import InterventionSchedule
    from ..intervention_schedule_compiler import InterventionExecutionContract
    from ..temperature import TemperatureScheduleProtocol

logger = logging.getLogger(__name__)


__all__ = [
    "DslAnalysis",
    "DslInitialConditionAdmission",
    "DslInitialConditionAssignment",
    "DslInitialConditionDeclaration",
    "DslParameterCoordinateUpdate",
    "DslSourceOrigin",
    "PreparedDslRequestSet",
    "derive_source_with_parameter_updates",
    "prepare_dsl_request_set",
]


# ------------------------------ normalized keys ------------------------------

_KEY_ALIASES: Dict[str, str] = {
    # temperature / states (keys are casefold-normalized)
    "t": "T",
    "degeneracy": "degeneracy",
    "kappa": "kappa",
    "c0": "C0",
    # energies / model params
    "dg_act": "dG_act",
    "ea": "Ea",
    "dg_eq": "dG_eq",
    "a": "A",
    "k": "k",
    "keq": "Keq",
    "kf": "kf",
    "kr": "kr",
    # units selector
    "energy": "energy",
}

_ARROW_RE = re.compile(r"<->|<=>|->|=>")
_TEMPERATURE_DIRECTIVE_PREFIXES = ("temp_step:", "temp_response:")
_SPECIES_TERM_RE = re.compile(
    r"^\s*(?:(-?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)\s*\*?\s*)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
_SCIENTIFIC_COEFFICIENT_MANTISSA_RE = re.compile(
    r"^\s*-?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))[eE]$"
)
_COMMA_SEMI_SPLIT_RE = re.compile(r"[,;]")
_SEMI_SPLIT_RE = re.compile(r"[;]")

_REACTION_KNOWN_KEYS = frozenset(
    {"kappa", "degeneracy", "kf", "k", "kr", "A", "Ea", "Keq", "dG_eq", "dG_act"}
)
# ------------------------------ data models ----------------------------------

@dataclass
class DSLIR:
    """
    Structured intermediate representation (IR) for the simulator DSL.

    Immutable analysis and Mechanism construction are derived from this IR to
    prevent semantic drift across entry points.
    """

    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    temperature_schedule: Optional["TemperatureScheduleProtocol"]
    intervention_schedule: Optional["InterventionSchedule"]
    steps: List[DslBuildStep]
    inline_initial_conditions: "DslInitialConditionAdmission"
    notes: List[str] = field(default_factory=list)


def _dsl_admission_payload(
    text: str,
    units: UnitsModel | None,
) -> dict[str, object]:
    if type(text) is not str:
        raise TypeError("DSL source must be an exact string.")
    if units is not None and type(units) is not UnitsModel:
        raise TypeError("DSL units must be UnitsModel or None.")
    effective = UnitsModel() if units is None else units
    return {
        "dsl": normalize_source_for_analysis(text),
        "energy_unit": effective.energy_unit,
        "temperature_K": effective.temperature_K,
        "standard_conc_M": effective.standard_conc_M,
    }


@dataclass(frozen=True, slots=True)
class DslSourceOrigin:
    authored_ordinal: int
    source_text: str
    physical_line_index: int = 0
    step_index: int | None = None
    coordinates: tuple[DslSourceCoordinate, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.authored_ordinal) is not int
            or self.authored_ordinal < 0
        ):
            raise ValueError("DSL origin ordinal must be a non-negative int.")
        if type(self.source_text) is not str:
            raise TypeError("DSL origin source text must be an exact string.")
        if type(self.physical_line_index) is not int or self.physical_line_index < 0:
            raise ValueError("DSL origin physical line index must be non-negative.")
        if self.step_index is not None and (
            type(self.step_index) is not int or self.step_index <= 0
        ):
            raise ValueError("DSL origin step index must be a positive int or None.")
        if type(self.coordinates) is not tuple or any(
            type(item) is not DslSourceCoordinate for item in self.coordinates
        ):
            raise TypeError(
                "DSL origin coordinates must be an exact tuple of DslSourceCoordinate values."
            )
        seen_coordinate_identities: set[tuple[str, str]] = set()
        for coordinate in self.coordinates:
            identity = (coordinate.canonical_name, coordinate.role)
            if identity in seen_coordinate_identities:
                raise ValueError("DSL source coordinate identities must be unique.")
            seen_coordinate_identities.add(identity)
            if coordinate.physical_line_index != self.physical_line_index:
                raise ValueError("DSL source coordinate line ownership is inconsistent.")
            if coordinate.value_end > len(self.source_text):
                raise ValueError("DSL source coordinate span exceeds exact source line text.")
            if not self.source_text[coordinate.value_start:coordinate.value_end].strip():
                raise ValueError("DSL source coordinate span must contain a value.")


@dataclass(frozen=True, slots=True)
class DslParameterCoordinateUpdate:
    """One exact writable coordinate value for typed DSL derivation."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError(
                "DSL parameter update name must be an exact nonempty string."
            )
        if type(self.value) not in {int, float}:
            raise TypeError("DSL parameter update value must be an exact number.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("DSL parameter update value must be finite.")
        object.__setattr__(self, "value", value)


_DSL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class DslInitialConditionAssignment:
    """One validated inline concentration assignment."""

    species: str
    value: float

    def __post_init__(self) -> None:
        if (
            type(self.species) is not str
            or not _DSL_IDENTIFIER_RE.fullmatch(self.species)
        ):
            raise ValueError(
                "Inline initial-condition species must be an identifier."
            )
        if type(self.value) not in {int, float}:
            raise TypeError(
                "Inline initial-condition values must be exact numbers."
            )
        value = float(self.value)
        try:
            value = validate_concentration(value, field="initial concentration")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class DslInitialConditionDeclaration:
    """One named block or one anonymous source-ordered declaration."""

    source_kind: str
    source_name: str | None
    assignments: tuple[DslInitialConditionAssignment, ...]

    def __post_init__(self) -> None:
        if self.source_kind not in {"named", "anonymous"}:
            raise ValueError(
                "Inline initial-condition kind must be named or anonymous."
            )
        if self.source_kind == "named":
            if (
                type(self.source_name) is not str
                or not _DSL_IDENTIFIER_RE.fullmatch(self.source_name)
            ):
                raise ValueError(
                    "Named initial-condition set names must be identifiers."
                )
        elif self.source_name is not None:
            raise ValueError(
                "Anonymous initial-condition declarations have no source name."
            )
        if type(self.assignments) is not tuple or any(
            type(item) is not DslInitialConditionAssignment
            for item in self.assignments
        ):
            raise TypeError(
                "Inline initial-condition assignments must be an exact tuple."
            )


@dataclass(frozen=True, slots=True)
class DslInitialConditionAdmission:
    """The sole typed inline-IC parser result for one exact source."""

    declarations: tuple[DslInitialConditionDeclaration, ...] = ()
    materialized_source: str = ""

    def __post_init__(self) -> None:
        if type(self.declarations) is not tuple or any(
            type(item) is not DslInitialConditionDeclaration
            for item in self.declarations
        ):
            raise TypeError(
                "Inline initial-condition declarations must be an exact tuple."
            )
        if type(self.materialized_source) is not str:
            raise TypeError(
                "Materialized inline initial-condition source must be text."
            )
        names = [
            declaration.source_name.casefold()
            for declaration in self.declarations
            if declaration.source_name is not None
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                "Inline initial-condition set names must be unique."
            )
        named_species: dict[str, set[str]] = {}
        anonymous_species: set[str] = set()
        for declaration in self.declarations:
            seen = (
                named_species.setdefault(
                    str(declaration.source_name).casefold(),
                    set(),
                )
                if declaration.source_kind == "named"
                else anonymous_species
            )
            for assignment in declaration.assignments:
                if assignment.species in seen:
                    scope = (
                        f"set {declaration.source_name!r}"
                        if declaration.source_kind == "named"
                        else "the anonymous initial-condition set"
                    )
                    raise ValueError(
                        "Duplicate inline initial-condition assignment for "
                        f"species {assignment.species!r} in {scope}."
                    )
                seen.add(assignment.species)

    @property
    def has_declarations(self) -> bool:
        return bool(self.declarations)

    @property
    def named_declarations(
        self,
    ) -> tuple[DslInitialConditionDeclaration, ...]:
        return tuple(
            item for item in self.declarations if item.source_kind == "named"
        )

@dataclass(frozen=True, slots=True)
class DslAnalysis:
    """Deeply immutable, source-backed public DSL admission evidence."""

    normalized_text: str
    origins: tuple[DslSourceOrigin, ...]
    species_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    inline_initial_conditions: DslInitialConditionAdmission
    temperature_K: float
    energy_unit: str

    def __post_init__(self) -> None:
        if type(self.normalized_text) is not str:
            raise TypeError("Normalized DSL text must be an exact string.")
        if type(self.origins) is not tuple or any(
            type(item) is not DslSourceOrigin for item in self.origins
        ):
            raise TypeError(
                "DSL origins must be an exact tuple of DslSourceOrigin values."
            )
        if tuple(item.authored_ordinal for item in self.origins) != tuple(
            range(len(self.origins))
        ):
            raise ValueError("DSL origin ordinals must be contiguous.")
        seen_coordinates: set[tuple[str, str]] = set()
        for origin in self.origins:
            for coordinate in origin.coordinates:
                identity = (coordinate.canonical_name, coordinate.role)
                if identity in seen_coordinates:
                    raise ValueError(
                        "DSL source coordinate identities must be unique."
                    )
                seen_coordinates.add(identity)
        for attr, label in (
            ("species_names", "species"),
            ("observable_names", "observable"),
        ):
            values = getattr(self, attr)
            if type(values) is not tuple or any(
                type(item) is not str
                or not item
                or item != item.strip()
                for item in values
            ):
                raise TypeError(
                    f"DSL {label} names must be an exact tuple of "
                    "nonempty strings."
                )
            if len(set(values)) != len(values):
                raise ValueError(f"DSL {label} names must be unique.")
        if set(self.species_names) & set(self.observable_names):
            raise ValueError(
                "DSL observable names must not duplicate model species names."
            )
        if type(self.inline_initial_conditions) is not DslInitialConditionAdmission:
            raise TypeError(
                "DSL inline initial conditions must be one typed admission."
            )
        if type(self.temperature_K) not in {int, float} or isinstance(
            self.temperature_K, bool
        ):
            raise TypeError("DSL analysis temperature_K must be numeric.")
        temperature_K = float(self.temperature_K)
        try:
            temperature_K = validate_temperature(
                temperature_K,
                field="DSL analysis temperature_K",
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        object.__setattr__(self, "temperature_K", temperature_K)
        if type(self.energy_unit) is not str:
            raise TypeError("DSL analysis energy_unit must be an exact string.")
        from .common import normalize_energy_unit

        try:
            energy_unit = normalize_energy_unit(self.energy_unit)
        except ValueError as exc:
            raise ValueError(
                f"DSL analysis energy_unit is invalid: {self.energy_unit!r}."
            ) from exc
        object.__setattr__(self, "energy_unit", energy_unit)


class PreparedDslRequestSet:
    """Single-parse owner of one immutable analysis and execution structure."""

    __slots__ = (
        "analysis",
        "default_units",
        "__mechanism",
        "__intervention_schedule",
        "__intervention_execution",
        "__parameter_algebra_spec",
        "__compiled_ode_topology",
        "__symbolic_jacobian_structure",
        "__symbolic_structure_initialized",
        "__structure_lock",
    )

    def __init__(
        self,
        *,
        analysis: DslAnalysis,
        default_units: UnitsModel,
        mechanism: object,
        intervention_schedule: InterventionSchedule | None,
        intervention_execution: InterventionExecutionContract | None,
        parameter_algebra_spec: object,
    ) -> None:
        from kindred.core.intervention_schedule import InterventionSchedule
        from kindred.core.intervention_schedule_compiler import (
            InterventionExecutionContract,
        )

        if type(analysis) is not DslAnalysis:
            raise TypeError("analysis must be DslAnalysis.")
        if intervention_schedule is not None and type(intervention_schedule) is not InterventionSchedule:
            raise TypeError("Parameterized intervention authority must be an exact InterventionSchedule.")
        if intervention_execution is not None and not isinstance(intervention_execution, InterventionExecutionContract):
            raise TypeError("Literal intervention authority must be an InterventionExecutionContract.")
        if intervention_schedule is not None:
            if not intervention_schedule.is_parameterized:
                raise ValueError("Only a parameterized intervention schedule may remain declarative on prepared DSL state.")
            if intervention_execution is not None:
                raise ValueError("Prepared DSL intervention authority cannot be both declarative and compiled.")
        elif intervention_execution is None:
            raise ValueError("Literal prepared DSL state requires its compiled intervention contract.")
        self.analysis = analysis
        self.default_units = default_units
        self.__mechanism = mechanism
        self.__intervention_schedule = intervention_schedule
        self.__intervention_execution = intervention_execution
        self.__parameter_algebra_spec = parameter_algebra_spec
        self.__compiled_ode_topology = None
        self.__symbolic_jacobian_structure = None
        self.__symbolic_structure_initialized = False
        self.__structure_lock = Lock()

    def mutable_mechanism_shell(self) -> object:
        from collections.abc import Mapping

        from kindred.core.mechanism import Mechanism

        def clone_metadata(value: object) -> object:
            if isinstance(value, Mapping):
                return {key: clone_metadata(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clone_metadata(item) for item in value]
            if isinstance(value, tuple):
                return tuple(clone_metadata(item) for item in value)
            if isinstance(value, (set, frozenset)):
                return type(value)(clone_metadata(item) for item in value)
            return value

        shell = Mechanism()
        shell.species = self.__mechanism.species
        shell.metadata = clone_metadata(
            getattr(self.__mechanism, "metadata", {}) or {}
        )
        shell.reactions = list(self.__mechanism.reactions)
        shell.equilibria = [
            replace(equilibrium, metadata=clone_metadata(equilibrium.metadata))
            for equilibrium in self.__mechanism.equilibria
        ]
        compiled_series = getattr(self.__mechanism, "_compiled_algebra_series", None)
        if compiled_series is not None:
            setattr(shell, "_compiled_algebra_series", compiled_series)
        return shell

    def accepted_mechanism_for_structure(self) -> object:
        """Return the privately retained accepted mechanism for read-only use."""

        return self.__mechanism

    def compiled_ode_topology(self) -> object:
        """Lazily retain one immutable ODE topology for this accepted owner."""

        topology = self.__compiled_ode_topology
        if topology is None:
            with self.__structure_lock:
                topology = self.__compiled_ode_topology
                if topology is None:
                    from kindred.core.ode_builder import CompiledOdeTopology

                    topology = CompiledOdeTopology.from_mechanism(
                        self.__mechanism
                    )
                    self.__compiled_ode_topology = topology
        return topology

    def symbolic_jacobian_structure(self) -> object:
        """Lazily retain one symbolic structure for this accepted mechanism."""

        structure = self.__symbolic_jacobian_structure
        if not self.__symbolic_structure_initialized:
            with self.__structure_lock:
                if not self.__symbolic_structure_initialized:
                    from kindred.core.symbolic.errors import (
                        UnsupportedSymbolicExpressionError,
                    )
                    from kindred.core.symbolic.jacobian import (
                        build_symbolic_jacobian_structure,
                    )

                    try:
                        structure = build_symbolic_jacobian_structure(
                            self.__mechanism
                        )
                    except UnsupportedSymbolicExpressionError:
                        structure = None
                    self.__symbolic_jacobian_structure = structure
                    self.__symbolic_structure_initialized = True
                structure = self.__symbolic_jacobian_structure
        return structure

    def temperature_schedule_for_execution(self) -> Optional["TemperatureScheduleProtocol"]:
        """Return the accepted mechanism's retained typed schedule directly."""

        from kindred.core.mechanism_metadata import MechanismMetadataKeys

        metadata = getattr(self.__mechanism, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return None
        return metadata.get(MechanismMetadataKeys.TEMPERATURE_SCHEDULE)

    def parameterized_intervention_schedule_for_binding(self) -> InterventionSchedule | None:
        return self.__intervention_schedule

    def authored_stop_declarations(self) -> tuple[str, ...]:
        from kindred.core.intervention_schedule import InterventionStopAction
        triggers = (
            self.__intervention_schedule.triggers
            if self.__intervention_schedule is not None
            else self.__intervention_execution.triggers
        )
        return tuple(item.source_text for item in triggers
                     if isinstance(item.then, InterventionStopAction) and item.source_text is not None)

    def intervention_execution_contract(self) -> InterventionExecutionContract | None:
        return self.__intervention_execution

    def parameter_algebra_spec_for_execution(self) -> object:
        return self.__parameter_algebra_spec


# ------------------------------ helpers --------------------------------------

def _norm_key(k: str) -> str:
    key = k.strip()
    normalized = key.casefold()
    if normalized == "keq":
        return "Keq"
    # Bare uppercase K is not lowercase k and is not accepted as a Keq alias.
    if key == "K":
        return "K"
    # Match the ASCII key table without changing existing key-case handling.
    return _KEY_ALIASES.get(normalized, key)


def _parse_keyvals(
    rest: str,
    *,
    reject_duplicate_canonical_keys: bool = False,
    line_number: int | None = None,
    line_content: str | None = None,
    coordinate_spans: list[tuple[str, int, int]] | None = None,
    source_offset: int = 0,
) -> Dict[str, str]:
    """
    Parse comma or semicolon-separated key=value pairs with lenient spacing.

    Example: "A=1.2e10, Ea=50, energy=kJ/mol" -> {"A":"1.2e10","Ea":"50","energy":"kJ/mol"}
    Example: "kf=1.5; kr=0.25" -> {"kf":"1.5","kr":"0.25"}
    """
    out: Dict[str, str] = {}
    original_spellings: Dict[str, str] = {}
    # Inline comments are source bytes, not part of the admitted key/value
    # grammar.  Keep offsets relative to the original line so exact Apply
    # can retain the comment and everything after it byte-for-byte.
    comment_index = rest.find("#")
    code = rest if comment_index < 0 else rest[:comment_index]
    if not code.strip():
        return out
    # Split on both commas and semicolons to support both formats
    cursor = 0
    for chunk in _COMMA_SEMI_SPLIT_RE.split(code):
        chunk_offset = cursor
        cursor += len(chunk)
        if cursor < len(code):
            cursor += 1
        if not chunk.strip():
            continue
        if "=" not in chunk:
            # tolerate stray tokens like "state=..." lines; raise for truly malformed
            raise invalid_keyvalue_pair_error(chunk)
        k, v = chunk.split("=", 1)
        raw_key = k.strip()
        canonical_key = _norm_key(raw_key)
        previous_spelling = original_spellings.get(canonical_key)
        if reject_duplicate_canonical_keys and previous_spelling is not None:
            raise _DSLError(
                f"Duplicate parameter: '{previous_spelling}' and '{raw_key}' both resolve to {canonical_key}",
                line_number=line_number,
                line_content=line_content,
            )
        original_spellings[canonical_key] = raw_key
        out[canonical_key] = v.strip()
        if coordinate_spans is not None and out[canonical_key]:
            value_start = (
                source_offset
                + chunk_offset
                + chunk.find("=")
                + 1
                + (len(v) - len(v.lstrip()))
            )
            value_end = value_start + len(out[canonical_key])
            coordinate_spans.append(
                (canonical_key, value_start, value_end)
            )
    return out


def _parse_species_side(text: str) -> Dict[str, float]:
    """
    Parse a side like "A + 2B + 0.5*C" into {name: coeff}.
    Coefficients default to 1.0 when omitted.
    """
    if not text.strip():
        return {}
    result: Dict[str, float] = {}
    terms: list[str] = []
    term_start = 0
    for index, character in enumerate(text):
        if character != "+":
            continue
        prefix = text[term_start:index]
        if _SCIENTIFIC_COEFFICIENT_MANTISSA_RE.fullmatch(prefix):
            exponent_start = index + 1
            exponent_end = exponent_start
            while (
                exponent_end < len(text)
                and text[exponent_end].isdigit()
            ):
                exponent_end += 1
            if exponent_end > exponent_start:
                continue
        terms.append(text[term_start:index])
        term_start = index + 1
    terms.append(text[term_start:])

    for term in terms:
        term = term.strip()
        if not term:
            raise _DSLError("Malformed stoichiometry: empty term (e.g., consecutive '+' signs)")
        m = _SPECIES_TERM_RE.match(term)
        if not m and "*" in term:
            # Allow patterns like "2 * B" while rejecting "A B"
            m = _SPECIES_TERM_RE.match(term.replace(" ", ""))
        if not m:
            raise invalid_species_term_error(term)
        coef_s, name = m.groups()
        if name == "t":
            raise _DSLError(
                "'t' is reserved for simulation time and cannot be used as a species name"
            )
        coef = _float_or_none(coef_s) if coef_s is not None else 1.0
        if coef is None or coef <= 0.0:
            raise _DSLError("Stoichiometric coefficients must be positive finite real numbers")
        result[name] = result.get(name, 0.0) + coef
    if not result:
        raise empty_stoichiometry_error()
    return result


def _parse_stoich(ar: str) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Parse "A + B -> C" or "2A <-> B + C" into reactants, products, arrow.
    Returns (reactants, products, arrow).
    """
    m = _ARROW_RE.search(ar)
    if not m:
        raise missing_arrow_error(ar)
    arrow = m.group(0)
    lhs = ar[: m.start()]
    rhs = ar[m.end() :]
    react = _parse_species_side(lhs)  # allow "2*B"
    prod = _parse_species_side(rhs)
    return react, prod, arrow


def _float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(s)
    except (TypeError, ValueError, OverflowError):
        logger.debug(f"Failed to convert '{s}' to float", exc_info=True)
        raise invalid_number_error(str(s), "value")
    if not math.isfinite(v):
        raise invalid_number_error(str(s), "value")
    return v


_DEGENERACY_LITERAL_RE = re.compile(r"\+?\d+")


def _parse_degeneracy(
    raw: object,
    *,
    line_number: int | None,
    line_content: str | None,
) -> int:
    """Admit only a positive integer literal for an Eyring degeneracy."""

    if type(raw) is bool:
        raise _DSLError(
            "degeneracy must be a positive integer literal, not a boolean",
            line_number=line_number,
            line_content=line_content,
        )
    if type(raw) is not str:
        raise _DSLError(
            "degeneracy must be a positive integer literal",
            line_number=line_number,
            line_content=line_content,
        )
    literal = raw.strip()
    if not _DEGENERACY_LITERAL_RE.fullmatch(literal):
        raise _DSLError(
            "degeneracy must be a positive integer literal",
            line_number=line_number,
            line_content=line_content,
        )
    value = int(literal)
    if value <= 0:
        raise _DSLError(
            "degeneracy must be a positive integer",
            line_number=line_number,
            line_content=line_content,
        )
    return value


def _validate_rate_or_K(
    value: Optional[float],
    name: str,
    *,
    line_number: int | None = None,
    line_content: str | None = None,
) -> None:
    if value is None:
        return
    if value is None or math.isnan(value) or math.isinf(value):
        raise _DSLError(
            f"'{name}' must be finite, got {value}",
            line_number=line_number,
            line_content=line_content,
        )
    try:
        if str(name) == "Keq":
            admit_keq(value, field=name)
        else:
            admit_rate(value, field=name, allow_zero=str(name) in {"k", "kf"})
    except ValueError as exc:
        raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc


def _validate_equilibrium_rate_authority_for_dsl(
    *,
    has_kf: bool,
    has_kr: bool,
    has_Keq: bool,
    has_dG_eq: bool,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    try:
        validate_equilibrium_rate_authority_flags(
            has_kf=has_kf,
            has_kr=has_kr,
            has_Keq=has_Keq,
            has_dG_eq=has_dG_eq,
        )
    except ValueError as exc:
        raise _DSLError(
            str(exc),
            suggestion="Provide kf=... with either kr=... or Keq=/dG_eq=..., but not both.",
            examples=[
                "equilibrium: A <-> B; kf=4.0; kr=2.0",
                "equilibrium: A <-> B; kf=10.0; Keq=5.0",
                "equilibrium: A <-> B; kf=10.0; dG_eq=-4.0",
            ],
            line_number=line_number,
            line_content=line_content,
        ) from exc


def _forward_rate_family_from_params(
    params: Mapping[str, str],
    *,
    line_number: int | None,
    line_content: str | None,
) -> str:
    """Select exactly one authored Constant/Arrhenius/Eyring family."""

    if "kf" in params and "k" in params:
        raise _DSLError(
            "Forward rate authority cannot contain both k and kf.",
            line_number=line_number,
            line_content=line_content,
        )
    families = {
        family
        for family, present in (
            ("Constant", any(key in params for key in ("kf", "k"))),
            ("Arrhenius", any(key in params for key in ("A", "Ea"))),
            ("Eyring", "dG_act" in params),
        )
        if present
    }
    if len(families) > 1:
        raise _DSLError(
            "Forward rate authority must use exactly one of Constant "
            "(k or kf), Arrhenius (A and Ea), or Eyring (dG_act).",
            line_number=line_number,
            line_content=line_content,
        )
    family = next(iter(families), "Eyring")
    if "kappa" in params and family != "Eyring":
        raise _DSLError(
            "Per-step kappa is valid only for an Eyring forward rate.",
            line_number=line_number,
            line_content=line_content,
        )
    return family


# ------------------------------ main parsing ---------------------------------

_NAMED_INITIAL_BLOCK_START_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{\s*(?P<body>.*?)\s*$"
)
_BRACKET_INITIAL_ASSIGNMENT_RE = re.compile(
    r"^\[(?P<species>[A-Za-z_][A-Za-z0-9_]*)\]\s*=\s*(?P<value>.+?)\s*$"
)


def _inline_initial_number(
    token: str,
    *,
    set_name: str | None,
    species: str,
    line_number: int,
    line_content: str,
) -> float:
    context = (
        f"set {set_name!r}, species {species!r}"
        if set_name is not None
        else f"anonymous set, species {species!r}"
    )
    try:
        value = float(str(token).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise _DSLError(
            f"Initial condition value for {context} must be numeric.",
            line_number=line_number,
            line_content=line_content,
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise _DSLError(
            f"Initial condition value for {context} must be finite and "
            "nonnegative.",
            line_number=line_number,
            line_content=line_content,
        )
    try:
        return validate_concentration(value, field=f"Initial condition value for {context}")
    except ValueError as exc:
        raise _DSLError(
            str(exc),
            line_number=line_number,
            line_content=line_content,
        ) from exc


def _inline_initial_assignments(
    code: str,
    *,
    set_name: str | None,
    line_number: int,
    line_content: str,
) -> tuple[DslInitialConditionAssignment, ...]:
    stripped = str(code).strip()
    lowered = stripped.lower()
    if lowered.startswith(("init:", "initial:")):
        _prefix, body = stripped.split(":", 1)
        raw_parts = body.split(",")
        if not body.strip() or any(not part.strip() for part in raw_parts):
            raise _DSLError(
                "Initial condition directive requires comma-separated "
                "species=value assignments.",
                line_number=line_number,
                line_content=line_content,
            )
        pairs: list[tuple[str, str]] = []
        for part in raw_parts:
            if part.count("=") != 1:
                raise _DSLError(
                    "Initial condition directive assignments must use exactly "
                    "one '='.",
                    line_number=line_number,
                    line_content=line_content,
                )
            species, value_token = part.split("=", 1)
            species = species.strip()
            if not _DSL_IDENTIFIER_RE.fullmatch(species):
                raise _DSLError(
                    f"Initial condition species {species!r} must be an "
                    "identifier.",
                    line_number=line_number,
                    line_content=line_content,
                )
            pairs.append((species, value_token))
    else:
        raw_parts = stripped.split(",")
        if any(not part.strip() for part in raw_parts):
            raise _DSLError(
                "Bracket initial conditions must be comma-separated "
                "assignments.",
                line_number=line_number,
                line_content=line_content,
            )
        pairs = []
        for part in raw_parts:
            match = _BRACKET_INITIAL_ASSIGNMENT_RE.fullmatch(part.strip())
            if match is None:
                raise _DSLError(
                    "Initial conditions must use '[Species] = value' or "
                    "'init: Species=value' syntax.",
                    line_number=line_number,
                    line_content=line_content,
                )
            pairs.append(
                (
                    str(match.group("species")),
                    str(match.group("value")),
                )
            )
    return tuple(
        DslInitialConditionAssignment(
            species=species,
            value=_inline_initial_number(
                value_token,
                set_name=set_name,
                species=species,
                line_number=line_number,
                line_content=line_content,
            ),
        )
        for species, value_token in pairs
    )


def _commented_initial_source(
    lines: list[str],
    claimed_indexes: set[int],
) -> str:
    rendered = [
        (
            INITIAL_PROVENANCE_PREFIX + line.rstrip()
            if index in claimed_indexes
            else line.rstrip()
        )
        for index, line in enumerate(lines)
    ]
    return "\n".join(rendered).strip()


def _admit_unique_initial_assignments(
    assignments: tuple[DslInitialConditionAssignment, ...],
    seen: set[str],
    *,
    scope: str,
    line_number: int,
    line_content: str,
) -> None:
    """Reject duplicate species before constructing the typed admission."""

    for assignment in assignments:
        if assignment.species in seen:
            raise _DSLError(
                "Duplicate inline initial-condition assignment for "
                f"species {assignment.species!r} in {scope}.",
                line_number=line_number,
                line_content=line_content,
            )
        seen.add(assignment.species)


def _admit_inline_initial_conditions(
    text: str,
) -> tuple[DslInitialConditionAdmission, frozenset[int]]:
    """Parse all inline IC syntax once and return claimed zero-based lines."""

    if type(text) is not str:
        raise TypeError("DSL source must be an exact string.")
    lines = text.splitlines()
    declarations: list[DslInitialConditionDeclaration] = []
    claimed_indexes: set[int] = set()
    named_keys: set[str] = set()
    anonymous_species: set[str] = set()
    index = 0
    while index < len(lines):
        raw = lines[index]
        code = raw.split("#", 1)[0].strip()
        if not code:
            index += 1
            continue
        header = _NAMED_INITIAL_BLOCK_START_RE.fullmatch(code)
        if header is not None:
            set_name = str(header.group("name"))
            body = str(header.group("body"))
            # Parameter algebra declarations own their own grammar.
            if set_name.lower() in {"let", "param"}:
                index += 1
                continue
            key = set_name.casefold()
            if key in named_keys:
                raise _DSLError(
                    f"Duplicate inline initial-condition set name: "
                    f"{set_name!r}.",
                    line_number=index + 1,
                    line_content=raw,
                )
            named_keys.add(key)
            start = index
            assignments: list[DslInitialConditionAssignment] = []
            named_species: set[str] = set()
            one_line_closed = False
            if "}" in body:
                before_close, close, after_close = body.partition("}")
                if close and after_close.strip():
                    raise _DSLError(
                        "Inline initial-condition block closing '}' must end "
                        "the declaration.",
                        line_number=index + 1,
                        line_content=raw,
                    )
                if before_close.strip():
                    parsed = _inline_initial_assignments(
                        before_close.strip(),
                        set_name=set_name,
                        line_number=index + 1,
                        line_content=raw,
                    )
                    _admit_unique_initial_assignments(
                        parsed,
                        named_species,
                        scope=f"set {set_name!r}",
                        line_number=index + 1,
                        line_content=raw,
                    )
                    assignments.extend(parsed)
                one_line_closed = True
            elif body.strip():
                parsed = _inline_initial_assignments(
                    body,
                    set_name=set_name,
                    line_number=index + 1,
                    line_content=raw,
                )
                _admit_unique_initial_assignments(
                    parsed,
                    named_species,
                    scope=f"set {set_name!r}",
                    line_number=index + 1,
                    line_content=raw,
                )
                assignments.extend(parsed)
            if one_line_closed:
                end = index
            else:
                cursor = index + 1
                while cursor < len(lines):
                    block_raw = lines[cursor]
                    block_code = block_raw.split("#", 1)[0].strip()
                    if not block_code:
                        cursor += 1
                        continue
                    if block_code == "}":
                        break
                    if "}" in block_code:
                        before_close, close, after_close = block_code.partition(
                            "}"
                        )
                        if not close or after_close.strip():
                            raise _DSLError(
                                "Inline initial-condition block closing '}' "
                                "must end the declaration.",
                                line_number=cursor + 1,
                                line_content=block_raw,
                            )
                        if before_close.strip():
                            parsed = _inline_initial_assignments(
                                before_close,
                                set_name=set_name,
                                line_number=cursor + 1,
                                line_content=block_raw,
                            )
                            _admit_unique_initial_assignments(
                                parsed,
                                named_species,
                                scope=f"set {set_name!r}",
                                line_number=cursor + 1,
                                line_content=block_raw,
                            )
                            assignments.extend(parsed)
                        break
                    parsed = _inline_initial_assignments(
                        block_code,
                        set_name=set_name,
                        line_number=cursor + 1,
                        line_content=block_raw,
                    )
                    _admit_unique_initial_assignments(
                        parsed,
                        named_species,
                        scope=f"set {set_name!r}",
                        line_number=cursor + 1,
                        line_content=block_raw,
                    )
                    assignments.extend(parsed)
                    cursor += 1
                if cursor >= len(lines):
                    raise _DSLError(
                        f"Inline initial-condition set {set_name!r} is "
                        "missing a closing '}'.",
                        line_number=start + 1,
                        line_content=raw,
                    )
                end = cursor
            claimed_indexes.update(range(start, end + 1))
            declarations.append(
                DslInitialConditionDeclaration(
                    source_kind="named",
                    source_name=set_name,
                    assignments=tuple(assignments),
                )
            )
            index = end + 1
            continue
        lowered = code.lower()
        if code.startswith("[") or lowered.startswith(("init:", "initial:")):
            parsed = _inline_initial_assignments(
                code,
                set_name=None,
                line_number=index + 1,
                line_content=raw,
            )
            _admit_unique_initial_assignments(
                parsed,
                anonymous_species,
                scope="the anonymous initial-condition set",
                line_number=index + 1,
                line_content=raw,
            )
            claimed_indexes.add(index)
            declarations.append(
                DslInitialConditionDeclaration(
                    source_kind="anonymous",
                    source_name=None,
                    assignments=parsed,
                )
            )
        index += 1
    return (
        DslInitialConditionAdmission(
            declarations=tuple(declarations),
            materialized_source=_commented_initial_source(
                lines,
                claimed_indexes,
            ),
        ),
        frozenset(claimed_indexes),
    )


def _admit_temperature_directive(text: str) -> str | None:
    """Admit zero or one top-level typed temperature directive."""
    directives: list[tuple[int, str, str]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        for prefix in _TEMPERATURE_DIRECTIVE_PREFIXES:
            if lower.startswith(prefix):
                directives.append((line_number, raw, prefix[:-1]))
                break
    if len(directives) > 1:
        line_number, raw, _kind = directives[1]
        kinds = ", ".join(kind for _number, _raw, kind in directives)
        raise _DSLError(
            "At most one top-level temperature schedule directive is "
            f"allowed; found {len(directives)} ({kinds}).",
            suggestion="Keep only one temp_step: or temp_response: directive.",
            line_number=line_number,
            line_content=raw,
        )
    return directives[0][1] if directives else None


def _line_content(raw_line: str) -> str:
    """Remove only the physical line ending, retaining authored whitespace."""

    if raw_line.endswith("\r\n"):
        return raw_line[:-2]
    if raw_line.endswith(("\r", "\n")):
        return raw_line[:-1]
    return raw_line


def _source_coordinates_from_key_value_spans(
    key_value_spans: tuple[tuple[str, int, int], ...],
    *,
    physical_line_index: int,
    step_index: int,
    step: DslBuildStep,
    line_content: str,
) -> tuple[DslSourceCoordinate, ...]:
    """Project parser-captured value spans into canonical source identities."""

    equilibrium = bool(
        step.is_equilibrium or step.reversible
    )
    coordinates: list[DslSourceCoordinate] = []
    seen_names: set[str] = set()
    for role, value_start, value_end in key_value_spans:
        if role in {"k", "kf"}:
            if equilibrium:
                canonical_role = "kf"
            else:
                canonical_role = "k"
        elif role in {"kr", "Keq"}:
            if not equilibrium:
                continue
            canonical_role = role
        elif role in {"A", "Ea", "dG_act", "dG_eq"}:
            if role == "dG_eq" and not equilibrium:
                continue
            canonical_role = role
        elif role == "degeneracy":
            if not isinstance(step.forward_model, EyringRateModel):
                continue
            canonical_role = role
        elif role == "kappa":
            # kappa is a physical Eyring coordinate only when explicitly
            # authored.  Its source span remains tied to the original key
            # spelling; only the canonical authority name is normalized.
            if not isinstance(step.forward_model, EyringRateModel):
                continue
            canonical_role = "kappa"
        else:
            continue
        canonical_name = canonical_source_coordinate_name(
            step_index,
            canonical_role,
        )
        if canonical_name in seen_names:
            raise _DSLError(
                f"Duplicate source coordinate {canonical_name!r}.",
                line_number=physical_line_index + 1,
                line_content=line_content,
            )
        seen_names.add(canonical_name)
        coordinates.append(
            DslSourceCoordinate(
                canonical_name=canonical_name,
                role=canonical_role,
                physical_line_index=physical_line_index,
                value_start=value_start,
                value_end=value_end,
            )
        )
    return tuple(coordinates)


def _parse_dsl_ir(text: str, *, units: UnitsModel | None = None) -> "DSLIR":
    """
    Parse simulator DSL into a structured intermediate representation (IR).

    This is the single source of truth for top-level DSL admission. Immutable
    analysis and Mechanism construction derive from this IR so consumers never
    parse source independently.
    """
    u = units or UnitsModel()
    energy_unit = u.energy_unit
    temperature_K = float(u.temperature_K)
    standard_conc_M = float(u.standard_conc_M)

    temperature_directive = _admit_temperature_directive(text)
    steps: List[DslBuildStep] = []
    inline_initial_conditions, inline_initial_line_indexes = (
        _admit_inline_initial_conditions(text)
    )
    notes: List[str] = []

    # Build the temperature schedule after resolving the nominal temperature.
    from ..temperature_dsl import parse_temperature_schedule
    from ..intervention_schedule import _admit_intervention_directives

    intervention_schedule = _admit_intervention_directives(text)

    raw_lines = text.splitlines(keepends=True)
    line_contents = [_line_content(raw_line) for raw_line in raw_lines]
    if not any(line.strip() and not line.strip().startswith("#") for line in line_contents):
        raise _DSLError("DSL text is empty. Provide at least one reaction or directive.")
    lines = [line.rstrip() for line in line_contents]

    # Resolve source-wide settings before lowering any reaction.  A directive
    # may appear anywhere in the source, but each global setting has exactly
    # one source owner and duplicate definitions fail before chemistry is
    # constructed.
    global_directive_lines: set[int] = set()
    global_directives: dict[str, tuple[int, str]] = {}
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        lower = line.lower()
        if not line or line.startswith("#"):
            continue
        key: str | None = None
        if lower.startswith("energy="):
            key = "energy"
        elif lower.startswith("c0="):
            key = "C0"
        elif lower.startswith("t="):
            key = "T"
        if key is None:
            continue
        assignments = _parse_keyvals(
            line,
            reject_duplicate_canonical_keys=True,
            line_number=line_no,
            line_content=raw,
        )
        if set(assignments) != {key}:
            raise _DSLError(
                f"Global {key}= must be the only assignment on its line.",
                suggestion="Put each global directive on a separate line.",
                line_number=line_no,
                line_content=raw,
            )
        if key in global_directives:
            raise _DSLError(
                f"Duplicate global {key} directive.",
                suggestion=f"Keep exactly one source-wide {key}= definition.",
                line_number=line_no,
                line_content=raw,
            )
        global_directives[key] = (line_no, raw)
        global_directive_lines.add(line_no)

    if "energy" in global_directives:
        line_no, raw = global_directives["energy"]
        try:
            energy_unit = _parse_energy_unit_directive(raw.strip())
        except _DSLError as exc:
            if exc.line_number is None:
                raise _DSLError(
                    str(exc.message),
                    suggestion=exc.suggestion,
                    examples=exc.examples,
                    line_number=line_no,
                    line_content=raw,
                ) from exc
            raise
        notes.append(f"energy unit set to {energy_unit}")
    if "C0" in global_directives:
        line_no, raw = global_directives["C0"]
        try:
            standard_conc_M = _parse_standard_conc_directive(raw.strip())
        except _DSLError as exc:
            if exc.line_number is None:
                raise _DSLError(
                    str(exc.message),
                    suggestion=exc.suggestion,
                    examples=exc.examples,
                    line_number=line_no,
                    line_content=raw,
                ) from exc
            raise
        notes.append(f"C0 set to {standard_conc_M:g} M")

    if "T" in global_directives:
        line_no, raw = global_directives["T"]
        try:
            values = _parse_keyvals(raw.strip(), reject_duplicate_canonical_keys=True)
            temperature_K = validate_temperature(float(values.get("T", "")), field="T")
        except (ValueError, _DSLError) as exc:
            raise _DSLError(
                str(exc), line_number=line_no, line_content=raw,
            ) from exc
        notes.append(f"T set to {temperature_K:g} K")

    temperature_schedule = (
        parse_temperature_schedule(
            temperature_directive, nominal_temperature=temperature_K,
        )
        if temperature_directive is not None
        else None
    )

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        lower = line.lower()

        if not line or line.startswith("#"):
            continue

        # Temperature schedule lines: parsed separately above
        if lower.startswith(_TEMPERATURE_DIRECTIVE_PREFIXES):
            continue
        if lower.startswith("intervention:"):
            continue

        # Source-wide settings were resolved before lowering.
        if line_no in global_directive_lines:
            continue

        if line_no - 1 in inline_initial_line_indexes:
            continue

        classification = classify_parameter_algebra_declaration(raw, line_number=line_no)
        if classification.kind == "let":
            continue

        if classification.kind == "param":
            continue

        if classification.kind == "invalid_step_key_identifier":
            target = classification.raw_name or raw.strip()
            raise _DSLError(
                f"{target!r} is a step-local DSL key and cannot be declared as a parameter or observable.",
                suggestion=(
                    "Use a canonical indexed mechanism parameter such as k1, kf1, kr1, or Keq1, "
                    "or choose a longer ordinary name."
                ),
                line_number=line_no,
                line_content=raw,
            )

        if classification.kind == "unsupported_bare_assignment":
            target = classification.raw_name or raw.strip()
            raise _DSLError(
                f"Bare algebra assignment {target!r} is not supported.",
                suggestion="Use 'let name = expr' or 'param name = expr'.",
                line_number=line_no,
                line_content=raw,
            )

        # reaction: line
        if lower.startswith("reaction:"):
            _, rest = line.split(":", 1)
            leading_offset = len(line_contents[line_no - 1]) - len(
                line_contents[line_no - 1].lstrip()
            )
            parsed_step = _parse_reaction_step(
                rest,
                energy_unit=energy_unit,
                C0=standard_conc_M,
                line_number=line_no,
                line_content=raw,
                physical_line_index=line_no - 1,
                step_index=len(steps) + 1,
                source_offset=leading_offset + len("reaction:"),
            )
            steps.append(parsed_step)
            continue

        # equilibrium: line
        if lower.startswith("equilibrium:"):
            _, rest = line.split(":", 1)
            leading_offset = len(line_contents[line_no - 1]) - len(
                line_contents[line_no - 1].lstrip()
            )
            parsed_step = _parse_equilibrium_step(
                rest,
                energy_unit=energy_unit,
                C0=standard_conc_M,
                line_number=line_no,
                line_content=raw,
                physical_line_index=line_no - 1,
                step_index=len(steps) + 1,
                source_offset=leading_offset + len("equilibrium:"),
            )
            steps.append(parsed_step)
            continue

        # Fallback: Try to parse as bare arrow syntax (no "reaction:" or "equilibrium:" prefix)
        if _ARROW_RE.search(line):
            leading_offset = len(line_contents[line_no - 1]) - len(
                line_contents[line_no - 1].lstrip()
            )
            parsed_step = _parse_bare_arrow_step(
                line,
                energy_unit=energy_unit,
                C0=standard_conc_M,
                line_number=line_no,
                line_content=raw,
                physical_line_index=line_no - 1,
                step_index=len(steps) + 1,
                source_offset=leading_offset,
            )
            steps.append(parsed_step)
            continue

        raise _DSLError(f"unrecognized line: {line!r}", line_number=line_no, line_content=raw)

    if temperature_schedule is not None:
        notes.append(f"Temperature schedule detected: {temperature_schedule}")
    if intervention_schedule is not None:
        notes.append("Intervention schedule detected")

    return DSLIR(
        energy_unit=str(energy_unit),
        temperature_K=float(temperature_K),
        standard_conc_M=float(standard_conc_M),
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
        steps=steps,
        inline_initial_conditions=inline_initial_conditions,
        notes=notes,
    )


# ------------------------------ Mechanism building ---------------------------


def _parse_energy_unit_directive(line: str) -> str:
    from .common import normalize_energy_unit

    kv = _parse_keyvals(line, reject_duplicate_canonical_keys=True)
    raw = kv.get("energy")
    if not raw:
        raise _DSLError("energy= directive requires a value (kJ/mol, kcal/mol, or J/mol)")
    try:
        return normalize_energy_unit(raw)
    except ValueError:
        raise _DSLError(
            f"energy must be 'kJ/mol', 'kcal/mol', or 'J/mol', got {raw!r}",
            examples=["energy=kJ/mol", "energy=kcal/mol", "energy=J/mol"],
        )


def _parse_standard_conc_directive(line: str) -> float:
    kv = _parse_keyvals(line, reject_duplicate_canonical_keys=True)
    raw = kv.get("C0")
    if not raw:
        raise _DSLError("C0= directive requires a numeric value")
    C0 = _float_or_none(raw)
    if C0 is None:
        raise _DSLError("C0 must be numeric")
    try:
        return validate_c0(C0, field="C0")
    except ValueError as exc:
        raise _DSLError(str(exc)) from exc


def _split_stoich_and_params(
    text: str,
    *,
    reject_duplicate_canonical_keys: bool = False,
    line_number: int | None = None,
    line_content: str | None = None,
    coordinate_spans: list[tuple[str, int, int]] | None = None,
    source_offset: int = 0,
) -> Tuple[str, Dict[str, str]]:
    stoich_part, *tail = _SEMI_SPLIT_RE.split(text, maxsplit=1)
    params = (
        _parse_keyvals(
            tail[0],
            reject_duplicate_canonical_keys=reject_duplicate_canonical_keys,
            line_number=line_number,
            line_content=line_content,
            coordinate_spans=coordinate_spans,
            source_offset=source_offset + len(stoich_part) + 1,
        )
        if tail
        else {}
    )
    return stoich_part, params


# ----- shared validation helpers (C1-C3) used by both step parsers ---------

def _reject_per_step_globals(
    params: Dict[str, str],
    *,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    """Reject per-reaction T= and energy= overrides."""
    if "T" in params:
        raise _DSLError(
            "Per-reaction T= is not supported; use a global T= directive.",
            line_number=line_number, line_content=line_content,
        )
    if "energy" in params:
        raise _DSLError(
            "Per-reaction energy= is not supported; use a global energy= directive",
            line_number=line_number, line_content=line_content,
        )


def _reject_unknown_params(
    params: Dict[str, str],
    known_keys: frozenset,
    step_label: str,
    *,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    """Reject parameter keys not in known_keys."""
    unknown = set(params) - known_keys
    if unknown:
        raise _DSLError(
            f"Unknown {step_label} parameter(s): {', '.join(sorted(unknown))}",
            line_number=line_number, line_content=line_content,
        )


def _parse_reaction_like_step(
    *,
    stoich_part: str,
    params: Dict[str, str],
    energy_unit: str,
    C0: float,
    missing_eyring_message: str,
    line_number: int | None,
    line_content: str | None,
    step_label: str = "reaction",
) -> DslBuildStep:
    from .kinetics import normalize_energy_to_J_per_mol

    try:
        react, prod, arrow = _parse_stoich(stoich_part)
    except _DSLError as exc:
        if exc.line_number is None:
            raise _DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                           line_number=line_number, line_content=line_content) from exc
        raise
    reversible = arrow in ("<->", "<=>")

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _REACTION_KNOWN_KEYS, step_label,
                           line_number=line_number, line_content=line_content)

    model = _forward_rate_family_from_params(
        params,
        line_number=line_number,
        line_content=line_content,
    )
    degeneracy = 1
    if "degeneracy" in params:
        if model != "Eyring" or "dG_act" not in params:
            raise _DSLError(
                "degeneracy is valid only with an Eyring dG_act forward rate",
                line_number=line_number,
                line_content=line_content,
            )
        degeneracy = _parse_degeneracy(
            params["degeneracy"],
            line_number=line_number,
            line_content=line_content,
        )
    kappa_raw = _float_or_none(params.get("kappa"))
    if kappa_raw is not None:
        kappa = kappa_raw
    else:
        kappa = 1.0
    n = float(
        directional_order_fraction(
            react,
        )
    )

    forward_rate_raw = params.get("kf") if "kf" in params else params.get("k")
    kf = _float_or_none(forward_rate_raw)
    kr = _float_or_none(params.get("kr"))
    has_forward_authority = any(
        key in params for key in ("kf", "k", "A", "Ea", "dG_act")
    )
    if reversible:
        _validate_equilibrium_rate_authority_for_dsl(
            has_kf=has_forward_authority,
            has_kr=("kr" in params),
            has_Keq=("Keq" in params),
            has_dG_eq=("dG_eq" in params),
            line_number=line_number,
            line_content=line_content,
        )
    if "kf" in params or "k" in params:
        _validate_rate_or_K(kf, "kf" if "kf" in params else "k",
                            line_number=line_number, line_content=line_content)
    if "kr" in params:
        _validate_rate_or_K(kr, "kr", line_number=line_number, line_content=line_content)
        if kf is not None and kf == 0.0:
            raise _DSLError(
                "Explicit kf+kr reversible steps require both directional rates to be positive.",
                line_number=line_number,
                line_content=line_content,
            )
    explicit_keq = _float_or_none(params.get("Keq"))
    if "Keq" in params:
        _validate_rate_or_K(explicit_keq, "Keq", line_number=line_number, line_content=line_content)

    def energy(key: str) -> float:
        value = _float_or_none(params.get(key))
        if value is None:
            raise ValueError(f"{key} must be numeric")
        return admit_energy_j_per_mol(
            normalize_energy_to_J_per_mol(value, energy_unit), field=key
        )

    try:
        if model == "Arrhenius":
            prefactor = _float_or_none(params.get("A"))
            if prefactor is None:
                raise ValueError("Arrhenius requires numeric A and Ea")
            forward_model = ArrheniusRateModel(A=prefactor, Ea_J_per_mol=energy("Ea"))
        elif model == "Eyring":
            forward_model = EyringRateModel(
                dG_act_J_per_mol=energy("dG_act"), kappa=kappa,
                molecularity=n, standard_conc_M=C0, degeneracy=degeneracy,
            )
        else:
            if kf is None:
                raise ValueError(missing_eyring_message)
            forward_model = ConstantRateModel(k=kf)
        delta_g = energy("dG_eq") if "dG_eq" in params else None
    except (TypeError, ValueError, OverflowError) as exc:
        raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc

    return DslBuildStep(
        reactants=tuple((str(name), float(value)) for name, value in react.items()),
        products=tuple((str(name), float(value)) for name, value in prod.items()),
        reversible=reversible,
        forward_model=forward_model,
        kr=kr,
        is_equilibrium=False,
        standard_conc_M=C0,
        dG_eq_J_per_mol=delta_g,
        Keq_input=explicit_keq,
    )


def _parse_reaction_step(
    rest: str,
    *,
    energy_unit: str,
    C0: float,
    line_number: int,
    line_content: str,
    physical_line_index: int = 0,
    step_index: int = 1,
    source_offset: int = 0,
) -> DslBuildStep:
    coordinate_spans: list[tuple[str, int, int]] = []
    stoich_part, params = _split_stoich_and_params(
        rest,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
        coordinate_spans=coordinate_spans,
        source_offset=source_offset,
    )
    parsed_step = _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        C0=C0,
        missing_eyring_message="Eyring step requires dG_act or explicit k",
        line_number=line_number,
        line_content=line_content,
    )
    return replace(
        parsed_step,
        source_coordinates=_source_coordinates_from_key_value_spans(
            tuple(coordinate_spans),
            physical_line_index=physical_line_index,
            step_index=step_index,
            step=parsed_step,
            line_content=line_content,
        ),
    )


def _parse_bare_arrow_step(
    line: str,
    *,
    energy_unit: str,
    C0: float,
    line_number: int,
    line_content: str,
    physical_line_index: int = 0,
    step_index: int = 1,
    source_offset: int = 0,
) -> DslBuildStep:
    coordinate_spans: list[tuple[str, int, int]] = []
    stoich_part, params = _split_stoich_and_params(
        line,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
        coordinate_spans=coordinate_spans,
        source_offset=source_offset,
    )
    parsed_step = _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        C0=C0,
        missing_eyring_message="Bare arrow syntax requires k (or kf), or dG_act for Eyring",
        line_number=line_number,
        line_content=line_content,
    )
    return replace(
        parsed_step,
        source_coordinates=_source_coordinates_from_key_value_spans(
            tuple(coordinate_spans),
            physical_line_index=physical_line_index,
            step_index=step_index,
            step=parsed_step,
            line_content=line_content,
        ),
    )


def _parse_equilibrium_step(
    rest: str,
    *,
    energy_unit: str,
    C0: float,
    line_number: int,
    line_content: str,
    physical_line_index: int = 0,
    step_index: int = 1,
    source_offset: int = 0,
) -> DslBuildStep:
    coordinate_spans: list[tuple[str, int, int]] = []
    stoich_part, params = _split_stoich_and_params(
        rest,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
        coordinate_spans=coordinate_spans,
        source_offset=source_offset,
    )
    try:
        _, _, arrow = _parse_stoich(stoich_part)
    except _DSLError as exc:
        if exc.line_number is None:
            raise _DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                           line_number=line_number, line_content=line_content) from exc
        raise
    if arrow not in ("<->", "<=>"):
        raise _DSLError(
            "equilibrium must use '<->' or '<=>'",
            examples=["equilibrium: A <-> B ; Keq=2.0 ; kf=10"],
            line_number=line_number, line_content=line_content,
        )
    parsed_step = _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        C0=C0,
        missing_eyring_message="Equilibrium requires kf or a complete forward rate family",
        step_label="equilibrium",
        line_number=line_number,
        line_content=line_content,
    )
    parsed_step = replace(parsed_step, is_equilibrium=True)
    return replace(
        parsed_step,
        source_coordinates=_source_coordinates_from_key_value_spans(
            tuple(coordinate_spans),
            physical_line_index=physical_line_index,
            step_index=step_index,
            step=parsed_step,
            line_content=line_content,
        ),
    )


def _source_origins(
    text: str,
    *,
    steps: tuple[DslBuildStep, ...] | list[DslBuildStep] = (),
    parameter_algebra_spec: object | None = None,
) -> tuple[DslSourceOrigin, ...]:
    """Retain exact admitted line origins and their typed value spans."""

    coordinates_by_line: dict[int, list[DslSourceCoordinate]] = {}
    step_by_line: dict[int, int] = {}
    for step_index, step in enumerate(steps, start=1):
        for coordinate in step.source_coordinates:
            coordinates_by_line.setdefault(
                coordinate.physical_line_index,
                [],
            ).append(coordinate)
            step_by_line[coordinate.physical_line_index] = step_index
    if parameter_algebra_spec is not None:
        for statement in getattr(parameter_algebra_spec, "param_statements", ()):
            coordinate = getattr(statement, "source_coordinate", None)
            if isinstance(coordinate, DslSourceCoordinate):
                coordinates_by_line.setdefault(
                    coordinate.physical_line_index,
                    [],
                ).append(coordinate)
    origins: list[DslSourceOrigin] = []
    for physical_line_index, raw_line in enumerate(text.splitlines(keepends=True)):
        source_line = _line_content(raw_line)
        stripped = source_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        origins.append(
            DslSourceOrigin(
                authored_ordinal=len(origins),
                source_text=source_line,
                physical_line_index=physical_line_index,
                step_index=step_by_line.get(physical_line_index),
                coordinates=tuple(
                    coordinates_by_line.get(physical_line_index, ())
                ),
            )
        )
    return tuple(origins)


def _mechanism_build_input(
    ir: DSLIR,
    *,
    initials: Mapping[str, float] | None,
):
    """Freeze the private parse accumulator into the sole builder input."""

    from kindred.core.simulator.dsl_build_input import DslMechanismBuildInput

    canonical_initials = {
        str(name): float(value)
        for name, value in dict(initials or {}).items()
    }

    return DslMechanismBuildInput(
        energy_unit=ir.energy_unit,
        temperature_K=ir.temperature_K,
        standard_conc_M=ir.standard_conc_M,
        temperature_schedule=ir.temperature_schedule,
        steps=tuple(ir.steps),
        initials=tuple(canonical_initials.items()),
    )


def prepare_dsl_request_set(
    text: str,
    *,
    initials: Mapping[str, float] | None = None,
    units: UnitsModel | None = None,
) -> PreparedDslRequestSet:
    """Admit source once and publish its one immutable typed analysis."""

    from kindred.core.intervention_schedule import (
        coerce_intervention_schedule,
    )
    from kindred.core.intervention_schedule_compiler import (
        compile_intervention_schedule,
    )
    from kindred.core.simulator.dsl_build import build_mechanism_from_ir
    from kindred.core.simulator.parameter_algebra_spec import (
        parse_parameter_algebra_spec_from_dsl_text,
    )
    from kindred.core.algebra.simulation_series import (
        compile_algebra_observables,
    )

    units = units or UnitsModel()
    payload = _dsl_admission_payload(text, units)
    normalized_text = str(payload["dsl"])
    ir = _parse_dsl_ir(text, units=units)
    intervention_schedule = coerce_intervention_schedule(
        ir.intervention_schedule
    )
    mechanism = build_mechanism_from_ir(
        _mechanism_build_input(
            ir,
            initials=initials,
        )
    )
    mechanism_namespace = build_namespace_from_mechanism(mechanism)
    parameter_algebra_spec = parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_namespace=mechanism_namespace,
    )
    from .parameter_algebra_spec import admit_parameter_evaluation_graph
    parameter_algebra_spec = admit_parameter_evaluation_graph(parameter_algebra_spec, mechanism)
    from .parameter_algebra import apply_parameter_algebra_spec_to_mechanism
    from .parameter_coordinates import build_parameter_target_map
    apply_parameter_algebra_spec_to_mechanism(
        parameter_algebra_spec, mechanism=mechanism,
        parameter_target_map=build_parameter_target_map(mechanism),
    )
    # The retained spec is the sole admitted Algebra program.  Compile the
    # observable partition from its retained ASTs; passing raw source here (or
    # in the builder) would reparse the same Algebra text and violate the
    # parse-once boundary.
    try:
        compiled_algebra_series = compile_algebra_observables(
            mechanism_namespace=mechanism_namespace,
            retained_spec=parameter_algebra_spec,
            species_names={str(item) for item in mechanism.species_names()},
        )
    except ValueError as exc:
        raise _DSLError(str(exc)) from exc
    setattr(mechanism, "_compiled_algebra_series", compiled_algebra_series)
    species_names = tuple(str(item) for item in mechanism.species_names())
    if intervention_schedule is not None:
        intervention_schedule.validate_species(species_names)
        from kindred.core.intervention_schedule import ObservableTriggerCondition, InterventionScheduleError
        observable_names = set(compiled_algebra_series.observable_names) if compiled_algebra_series is not None else set()
        for trigger in intervention_schedule.triggers:
            if isinstance(trigger.condition, ObservableTriggerCondition) and trigger.condition.name not in observable_names:
                raise InterventionScheduleError(
                    f"Unknown stopping observable: {trigger.condition.name}",
                    operation="trigger", declaration_ordinal=trigger.declaration_ordinal, field="watch_observable",
                )
    parameterized_intervention_schedule = (
        intervention_schedule
        if intervention_schedule is not None
        and intervention_schedule.is_parameterized
        else None
    )
    intervention_execution = (
        None
        if parameterized_intervention_schedule is not None
        else compile_intervention_schedule(intervention_schedule)
    )
    analysis = DslAnalysis(
        normalized_text=normalized_text,
        origins=_source_origins(
            text,
            steps=ir.steps,
            parameter_algebra_spec=parameter_algebra_spec,
        ),
        species_names=species_names,
        observable_names=tuple(
            sorted(
                str(name)
                for name in parameter_algebra_spec.observable_names
                if str(name).strip()
            )
        ),
        inline_initial_conditions=ir.inline_initial_conditions,
        temperature_K=float(ir.temperature_K),
        energy_unit=str(ir.energy_unit),
    )
    metadata = getattr(mechanism, "metadata", None)
    if isinstance(metadata, dict):
        metadata["parameter_algebra_spec"] = parameter_algebra_spec
    return PreparedDslRequestSet(
        analysis=analysis,
        default_units=units,
        mechanism=mechanism,
        intervention_schedule=parameterized_intervention_schedule,
        intervention_execution=intervention_execution,
        parameter_algebra_spec=parameter_algebra_spec,
    )


def _analysis_source_coordinates(
    analysis: DslAnalysis,
) -> tuple[tuple[DslSourceCoordinate, DslSourceOrigin], ...]:
    """Return each admitted coordinate with its retained line owner."""

    pairs = tuple(
        (coordinate, origin)
        for origin in analysis.origins
        for coordinate in origin.coordinates
    )
    identities = [
        (coordinate.canonical_name, coordinate.role)
        for coordinate, _origin in pairs
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Retained DSL source coordinates are ambiguous.")
    return pairs


def derive_source_with_parameter_updates(
    source_text: str,
    base: PreparedDslRequestSet,
    updates: tuple[DslParameterCoordinateUpdate, ...],
) -> str:
    """Authenticate typed coordinates and rewrite the visible source bytes."""

    if base.analysis.inline_initial_conditions.has_declarations:
        raise ValueError(
            "Typed parameter derivation requires canonical materialized "
            "initial conditions."
        )
    if not updates:
        raise ValueError("Typed DSL parameter derivation requires updates.")
    source = MechanismAuthoringSource.from_parts(reactions_text=source_text)
    if not source_matches_analysis(source, base.analysis):
        raise ValueError(
            "Typed parameter derivation source does not match its retained analysis."
        )
    names = tuple(item.name for item in updates)
    if len(names) != len(set(names)):
        raise ValueError("DSL parameter update names must be unique.")

    spec = base.parameter_algebra_spec_for_execution()
    mechanism = base.accepted_mechanism_for_structure()
    from .step_indexing import lookup_source_coordinate_target
    from ..equilibrium_rate_authority import step_entry_authored_role_is_editable
    raw_lines = source_text.splitlines(keepends=True)
    source_lines = [_line_content(line) for line in raw_lines]
    line_endings = [
        line[len(content):]
        for line, content in zip(raw_lines, source_lines)
    ]
    if not raw_lines and source_text:
        source_lines = [source_text]
        line_endings = [""]

    # Coordinates retain their physical line index and exact half-open span.
    # Apply accepts only that retained location, and only when the current line
    # still matches admission byte-for-byte (apart from its line ending).
    for origin in base.analysis.origins:
        index = int(origin.physical_line_index)
        if index >= len(source_lines) or source_lines[index] != origin.source_text:
            raise ValueError(
                "Typed parameter derivation source line no longer matches admission."
            )
    coordinate_pairs = _analysis_source_coordinates(base.analysis)
    source_coordinates = {
        coordinate.canonical_name: (coordinate, origin)
        for coordinate, origin in coordinate_pairs
    }
    if len(source_coordinates) != len(coordinate_pairs):
        raise ValueError("Retained DSL source coordinates are ambiguous.")
    replacements: dict[int, list[tuple[int, int, str]]] = {}

    def retain_replacement(
        coordinate: DslSourceCoordinate,
        replacement: str,
    ) -> None:
        replacements.setdefault(int(coordinate.physical_line_index), []).append(
            (coordinate.value_start, coordinate.value_end, replacement)
        )

    for update in updates:
        coordinate_entry = source_coordinates.get(update.name)
        if coordinate_entry is None:
            raise LookupError(
                f"Parameter {update.name!r} is absent from retained DSL."
            )
        coordinate, _origin = coordinate_entry
        try:
            assignment_kind = spec.parameter_assignment_kind(update.name)
        except KeyError:
            assignment_kind = None
        # Classify from the admitted Algebra dependency map before inspecting
        # the coordinate role.  Algebra-owned assignments are never rewritten
        # as if their source spelling were an independent mechanism value.
        if assignment_kind in {"derived scalar", "mechanism-bound"}:
            raise ValueError(
                f"Scalar parameter {update.name!r} is not writable."
            )
        if assignment_kind == "independent scalar":
            if coordinate.role != "param":
                raise ValueError(
                    f"Scalar parameter {update.name!r} is not writable."
                )
            if not coordinate.capabilities.continuous:
                raise ValueError(
                    f"Parameter {update.name!r} is not a continuous writable coordinate."
                )
            retain_replacement(coordinate, repr(float(update.value)))
            continue
        if coordinate.role == "param":
            raise LookupError(
                f"Parameter {update.name!r} is absent from admitted Algebra assignments."
            )
        if not coordinate.capabilities.continuous:
            raise ValueError(
                f"Parameter {update.name!r} is not a continuous writable coordinate."
            )
        source_target = lookup_source_coordinate_target(mechanism, update.name)
        if source_target is None:
            raise ValueError(
                f"Parameter {update.name!r} has no admitted source owner."
            )
        kind, index, role, entry = source_target
        if str(role) != coordinate.role:
            raise ValueError(
                f"Parameter {update.name!r} has inconsistent admitted source role."
            )
        if bool(entry.get("locked", False)):
            raise ValueError(f"Parameter {update.name!r} is locked and not writable.")
        if (
            str(kind) == "equilibrium"
            and coordinate.capabilities.equilibrium_authority
            and not step_entry_authored_role_is_editable(entry, str(role))
        ):
            raise ValueError(f"Parameter {update.name!r} is a derived readout, not writable.")
        # Physical and authored indexed source coordinates are already in the
        # exact units and representation expected by their retained span.
        retain_replacement(coordinate, repr(float(update.value)))

    for line_index, line_replacements in replacements.items():
        seen_spans: set[tuple[int, int]] = set()
        ordered_replacements = sorted(
            line_replacements,
            key=lambda item: item[0],
            reverse=True,
        )
        previous_start = len(source_lines[line_index]) + 1
        for start, end, _replacement in ordered_replacements:
            if (start, end) in seen_spans:
                raise ValueError("Duplicate DSL source replacement span.")
            if not 0 <= start < end <= len(source_lines[line_index]):
                raise ValueError("DSL source replacement span is out of bounds.")
            if end > previous_start:
                raise ValueError("Overlapping DSL source replacement spans.")
            seen_spans.add((start, end))
            previous_start = start
        for start, end, replacement in ordered_replacements:
            source_lines[line_index] = (
                source_lines[line_index][:start]
                + replacement
                + source_lines[line_index][end:]
            )
    updated_text = "".join(
        content + ending
        for content, ending in zip(source_lines, line_endings)
    )
    return updated_text
