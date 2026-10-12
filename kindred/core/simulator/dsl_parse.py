"""
Computational Simulator DSL tools.

Current contract
----------------
- Steps use unprefixed reaction lines.
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

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from threading import Lock
from typing import Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import re

from ..equilibrium_rate_authority import (
    validate_equilibrium_rate_authority_flags,
)
from ..mechanism_source import (
    INITIAL_PROVENANCE_PREFIX,
)
from ..kinetics import ArrheniusRateModel, ConstantRateModel, EyringRateModel
from ..units import UnitsModel
from ..units import (
    admit_arrhenius_prefactor,
    admit_energy_j_per_mol,
    admit_keq,
    admit_rate,
    admit_role_value,
    energy_to_j_per_mol,
    normalize_energy_unit,
)
from ..species import validate_c0, validate_concentration, validate_temperature
from ..validation import six_digit_source_text
from ..stoichiometry import directional_order_fraction
from .errors import (
    DSLError as _DSLError,
    invalid_number_error,
    missing_arrow_error,
    invalid_species_term_error,
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
    from ..temperature import TemperatureSchedule


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
    temperature_schedule: Optional["TemperatureSchedule"]
    intervention_schedule: Optional["InterventionSchedule"]
    steps: List[DslBuildStep]
    inline_initial_conditions: "DslInitialConditionAdmission"
    # (line number, authored text) of each intervention declaration, by ordinal.
    intervention_lines: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class DslSourceOrigin:
    authored_ordinal: int
    source_text: str
    physical_line_index: int = 0
    step_index: int | None = None
    # The parser assigns each line's own unique coordinates.
    coordinates: tuple[DslSourceCoordinate, ...] = ()


@dataclass(frozen=True, slots=True)
class DslParameterCoordinateUpdate:
    """One exact writable coordinate value for typed DSL derivation."""

    name: str
    value: float


_DSL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class DslInitialConditionAssignment:
    """One validated inline concentration assignment."""

    species: str
    value: float


@dataclass(frozen=True, slots=True)
class DslInitialConditionDeclaration:
    """One named block or one anonymous source-ordered declaration."""

    source_kind: str
    source_name: str | None
    assignments: tuple[DslInitialConditionAssignment, ...]


@dataclass(frozen=True, slots=True)
class DslInitialConditionAdmission:
    """The sole typed inline-IC parser result for one exact source."""

    declarations: tuple[DslInitialConditionDeclaration, ...] = ()
    materialized_source: str = ""

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

    origins: tuple[DslSourceOrigin, ...]
    species_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    inline_initial_conditions: DslInitialConditionAdmission
    temperature_K: float
    standard_conc_M: float
    energy_unit: str
    # (line number, authored text) of each intervention declaration, by ordinal.
    intervention_lines: tuple[tuple[int, str], ...]


class PreparedDslRequestSet:
    """Single-parse owner of one immutable analysis and execution structure."""

    __slots__ = (
        "analysis",
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
        mechanism: object,
        intervention_schedule: InterventionSchedule | None,
        intervention_execution: InterventionExecutionContract | None,
        parameter_algebra_spec: object,
    ) -> None:
        # Its sole producer retains a parameterized schedule declaratively
        # and compiles every other schedule, including none.
        self.analysis = analysis
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
        shell.metadata = clone_metadata(self.__mechanism.metadata)
        shell.reactions = list(self.__mechanism.reactions)
        shell.equilibria = [
            replace(equilibrium, metadata=clone_metadata(equilibrium.metadata))
            for equilibrium in self.__mechanism.equilibria
        ]
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

    def temperature_schedule_for_execution(self) -> Optional["TemperatureSchedule"]:
        """Return the accepted mechanism's retained typed schedule directly."""

        from kindred.core.mechanism_metadata import MechanismMetadataKeys

        return self.__mechanism.metadata.get(MechanismMetadataKeys.TEMPERATURE_SCHEDULE)

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
            # every chunk must be a key=value pair
            raise _DSLError(
                f"Expected key=value, got '{chunk.strip()}'.",
                line_number=line_number,
                line_content=line_content,
            )
        k, v = chunk.split("=", 1)
        raw_key = k.strip()
        canonical_key = _norm_key(raw_key)
        previous_spelling = original_spellings.get(canonical_key)
        if reject_duplicate_canonical_keys and previous_spelling is not None:
            raise _DSLError(
                f"{raw_key} appears twice."
                if previous_spelling == raw_key
                else f"'{previous_spelling}' and '{raw_key}' both set {canonical_key}.",
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


def _ambiguous_stoichiometry_error(text: str) -> _DSLError:
    return _DSLError(
        f"Ambiguous stoichiometry {text.strip()!r}: separate coefficients from "
        "species with spaces, as in '2E + 2S' or '2 E2S', or write a "
        "scientific-notation coefficient with '*', as in '2e2*S'."
    )


def _parse_species_side(text: str) -> Dict[str, float]:
    """
    Parse a side like "A + 2B + 0.5*C" into {name: coeff}.
    Coefficients default to 1.0 when omitted.
    """
    if not text.strip():
        return {}
    result: Dict[str, Fraction] = {}
    terms: list[str] = []
    term_start = 0
    for index, character in enumerate(text):
        if character != "+":
            continue
        prefix = text[term_start:index]
        if (
            _SCIENTIFIC_COEFFICIENT_MANTISSA_RE.fullmatch(prefix)
            and text[index + 1:index + 2].isdigit()
        ):
            raise _ambiguous_stoichiometry_error(text)
        terms.append(text[term_start:index])
        term_start = index + 1
    terms.append(text[term_start:])

    for term in terms:
        term = term.strip()
        if not term:
            raise _DSLError("Malformed stoichiometry: empty term, for example consecutive '+' signs.")
        m = _SPECIES_TERM_RE.match(term)
        if not m and "*" in term:
            # Allow patterns like "2 * B" while rejecting "A B"
            m = _SPECIES_TERM_RE.match(term.replace(" ", ""))
        if not m:
            raise invalid_species_term_error(term)
        coef_s, name = m.groups()
        if (
            coef_s is not None
            and "e" in coef_s.casefold()
            and m.string is term
            and m.end(1) == m.start(2)
        ):
            raise _ambiguous_stoichiometry_error(text)
        if name == "t":
            raise _DSLError(
                "'t' is reserved for simulation time and cannot be used as a species name."
            )
        coef = _float_or_none(coef_s, "coefficient") if coef_s is not None else 1.0
        if coef <= 0.0:
            raise _DSLError("Stoichiometric coefficients must be positive.")
        exact = Fraction(1) if coef_s is None else Fraction(coef_s)
        result[name] = result.get(name, Fraction(0)) + exact
    return {name: float(total) for name, total in result.items()}


def _parse_stoich(ar: str) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Parse "A + B -> C" or "2A <-> B + C" into reactants, products, arrow.
    Returns (reactants, products, arrow).
    """
    m = _ARROW_RE.search(ar)
    if not m:
        raise missing_arrow_error()
    arrow = m.group(0)
    lhs = ar[: m.start()]
    rhs = ar[m.end() :]
    react = _parse_species_side(lhs)  # allow "2*B"
    prod = _parse_species_side(rhs)
    return react, prod, arrow


def _float_or_none(s: Optional[str], field: str) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(s)
    except (TypeError, ValueError, OverflowError):
        raise invalid_number_error(str(s), field)
    if not math.isfinite(v):
        raise invalid_number_error(str(s), field)
    return v


_DEGENERACY_LITERAL_RE = re.compile(r"\+?\d+")


def _parse_degeneracy(
    raw: str,
    *,
    line_number: int | None,
    line_content: str | None,
) -> int:
    """Admit only a positive integer literal for an Eyring degeneracy."""

    literal = raw.strip()
    if not _DEGENERACY_LITERAL_RE.fullmatch(literal) or int(literal) <= 0:
        raise _DSLError(
            "degeneracy must be a positive integer.",
            line_number=line_number,
            line_content=line_content,
        )
    return int(literal)


def _validate_rate_or_K(
    value: Optional[float],
    name: str,
    *,
    line_number: int | None = None,
    line_content: str | None = None,
) -> None:
    if value is None:
        return
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
            line_number=line_number,
            line_content=line_content,
        ) from exc


def _forward_rate_family_from_params(
    params: Mapping[str, str],
    *,
    reversible: bool,
    line_number: int | None,
    line_content: str | None,
) -> str:
    """Select exactly one authored Constant/Arrhenius/Eyring family."""

    if "kf" in params and "k" in params:
        raise _DSLError(
            "A step cannot have both k and kf.",
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
    if not families:
        raise _DSLError(
            "This reversible step has no forward rate. Add kf=, A= with Ea=, or "
            "dG_act=, plus one of kr=, Keq=, or dG_eq=."
            if reversible
            else "This step has no rate. Add k=, A= with Ea=, or dG_act=.",
            line_number=line_number,
            line_content=line_content,
        )
    family = next(iter(families))
    if family == "Arrhenius" and not ("A" in params and "Ea" in params):
        raise _DSLError(
            "An Arrhenius rate needs both A and Ea.",
            line_number=line_number,
            line_content=line_content,
        )
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


def _admit_temperature_directive(text: str) -> tuple[int, str] | None:
    """Admit zero or one top-level typed temperature directive and its line."""
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
            f"allowed. Found {len(directives)} ({kinds}).",
            line_number=line_number,
            line_content=raw,
        )
    return directives[0][:2] if directives else None


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

    equilibrium = step.reversible
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

    # Build the temperature schedule after resolving the nominal temperature.
    from ..temperature_dsl import TemperatureDSLError, parse_temperature_schedule
    from ..intervention_schedule import _admit_intervention_directives

    intervention_schedule, intervention_lines = _admit_intervention_directives(text)

    raw_lines = text.splitlines(keepends=True)
    line_contents = [_line_content(raw_line) for raw_line in raw_lines]
    lines = [line.rstrip() for line in line_contents]

    # Resolve source-wide settings before lowering any reaction.  A directive
    # may appear anywhere in the source, but each global setting has exactly
    # one source owner and duplicate definitions fail before chemistry is
    # constructed.
    global_directive_lines: set[int] = set()
    global_directives: dict[str, tuple[int, str]] = {}
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Only capital T sets temperature; lowercase t is simulation time.
        directive = re.match(r"(?:(?i:energy|c0)|T)\s*=(?!>)", line)
        if directive is None:
            continue
        key = _KEY_ALIASES[directive.group(0).split("=")[0].strip().lower()]
        assignments = _parse_keyvals(
            line,
            reject_duplicate_canonical_keys=True,
            line_number=line_no,
            line_content=raw,
        )
        if set(assignments) != {key}:
            raise _DSLError(
                f"Global {key}= must be the only assignment on its line.",
                line_number=line_no,
                line_content=raw,
            )
        if key in global_directives:
            raise _DSLError(
                f"Duplicate global {key} directive.",
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
                    line_number=line_no,
                    line_content=raw,
                ) from exc
            raise
    if "C0" in global_directives:
        line_no, raw = global_directives["C0"]
        try:
            standard_conc_M = _parse_standard_conc_directive(raw.strip())
        except _DSLError as exc:
            if exc.line_number is None:
                raise _DSLError(
                    str(exc.message),
                    line_number=line_no,
                    line_content=raw,
                ) from exc
            raise

    if "T" in global_directives:
        line_no, raw = global_directives["T"]
        try:
            values = _parse_keyvals(raw.strip(), reject_duplicate_canonical_keys=True)
            temperature_K = validate_temperature(_float_or_none(values["T"], "T"), field="T")
        except (ValueError, _DSLError) as exc:
            raise _DSLError(
                str(exc), line_number=line_no, line_content=raw,
            ) from exc

    temperature_schedule = None
    if temperature_directive is not None:
        line_no, raw = temperature_directive
        try:
            temperature_schedule = parse_temperature_schedule(
                raw, nominal_temperature=temperature_K,
            )
        except TemperatureDSLError as exc:
            raise _DSLError(
                str(exc), line_number=line_no, line_content=raw,
            ) from exc

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
                f"{target!r} is a reaction-line field, so it cannot name a param or let line.",
                line_number=line_no,
                line_content=raw,
            )

        if classification.kind == "unsupported_bare_assignment":
            target = classification.raw_name or raw.strip()
            raise _DSLError(
                f"Assignment {target!r} must start with let or param.",
                line_number=line_no,
                line_content=raw,
            )

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

        raise _DSLError(
            "Unrecognized line."
            + (" Temperature is set with capital T=." if re.match(r"t\s*=(?!>)", line) else ""),
            line_number=line_no,
            line_content=raw,
        )

    if not steps:
        raise _DSLError("Enter at least one reaction.")
    return DSLIR(
        energy_unit=str(energy_unit),
        temperature_K=float(temperature_K),
        standard_conc_M=float(standard_conc_M),
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
        steps=steps,
        inline_initial_conditions=inline_initial_conditions,
        intervention_lines=intervention_lines,
    )


# ------------------------------ Mechanism building ---------------------------


def _parse_energy_unit_directive(line: str) -> str:
    raw = _parse_keyvals(line, reject_duplicate_canonical_keys=True)["energy"]
    try:
        return normalize_energy_unit(raw)
    except ValueError:
        raise _DSLError(
            f"energy must be kJ/mol, kcal/mol, or J/mol, not '{raw}'."
        )


def _parse_standard_conc_directive(line: str) -> float:
    raw = _parse_keyvals(line, reject_duplicate_canonical_keys=True)["C0"]
    try:
        return validate_c0(_float_or_none(raw, "C0"), field="C0")
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
            "T= cannot be set on a reaction line. Write T= on its own line.",
            line_number=line_number, line_content=line_content,
        )
    if "energy" in params:
        raise _DSLError(
            "energy= cannot be set on a reaction line. Write energy= on its own line.",
            line_number=line_number, line_content=line_content,
        )


def _reject_unknown_params(
    params: Dict[str, str],
    known_keys: frozenset,
    *,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    """Reject parameter keys not in known_keys."""
    unknown = set(params) - known_keys
    if unknown:
        raise _DSLError(
            f"Unknown reaction field{'s' * (len(unknown) > 1)}: {', '.join(sorted(unknown))}.",
            line_number=line_number, line_content=line_content,
        )


def _parse_reaction_like_step(
    *,
    stoich_part: str,
    params: Dict[str, str],
    energy_unit: str,
    C0: float,
    line_number: int | None,
    line_content: str | None,
) -> DslBuildStep:
    react, prod, arrow = _parse_stoich(stoich_part)
    reversible = arrow in ("<->", "<=>")
    if reversible and not (react and prod):
        raise _DSLError(
            "A reversible step needs species on both sides.",
            line_number=line_number,
            line_content=line_content,
        )
    if not reversible and react == prod:
        raise _DSLError(
            "This step leaves every species unchanged.",
            line_number=line_number,
            line_content=line_content,
        )

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _REACTION_KNOWN_KEYS,
                           line_number=line_number, line_content=line_content)
    reverse_only = [key for key in ("kr", "Keq", "dG_eq") if key in params]
    if not reversible and reverse_only:
        raise _DSLError(
            f"An irreversible step cannot have {', '.join(reverse_only)}. "
            "Use <-> for a reversible step, or remove the field.",
            line_number=line_number,
            line_content=line_content,
        )

    model = _forward_rate_family_from_params(
        params,
        reversible=reversible,
        line_number=line_number,
        line_content=line_content,
    )
    degeneracy = 1
    if "degeneracy" in params:
        if model != "Eyring" or "dG_act" not in params:
            raise _DSLError(
                "degeneracy is valid only with an Eyring dG_act forward rate.",
                line_number=line_number,
                line_content=line_content,
            )
        degeneracy = _parse_degeneracy(
            params["degeneracy"],
            line_number=line_number,
            line_content=line_content,
        )
    kappa = _float_or_none(params["kappa"], "kappa") if "kappa" in params else 1.0
    n = float(
        directional_order_fraction(
            react,
        )
    )

    forward_rate_raw = params.get("kf") if "kf" in params else params.get("k")
    kf = _float_or_none(forward_rate_raw, "kf" if "kf" in params else "k")
    kr = _float_or_none(params.get("kr"), "kr")
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
    explicit_keq = _float_or_none(params.get("Keq"), "Keq")
    if "Keq" in params:
        _validate_rate_or_K(explicit_keq, "Keq", line_number=line_number, line_content=line_content)

    def energy(key: str) -> float:
        return admit_energy_j_per_mol(
            energy_to_j_per_mol(_float_or_none(params[key], key), energy_unit), field=key
        )

    try:
        if model == "Arrhenius":
            prefactor = _float_or_none(params["A"], "A")
            activation = energy("Ea")
            forward_model = ArrheniusRateModel(
                A=admit_arrhenius_prefactor(prefactor, field="Arrhenius A"), Ea_J_per_mol=activation,
            )
        elif model == "Eyring":
            activation = energy("dG_act")
            forward_model = EyringRateModel(
                dG_act_J_per_mol=activation,
                kappa=admit_role_value(kappa, role="kappa", unit="1", allow_exact_zero=False, field="Eyring kappa"),
                molecularity=n, standard_conc_M=C0, degeneracy=degeneracy,
            )
        else:
            forward_model = ConstantRateModel(k=kf)
        delta_g = energy("dG_eq") if "dG_eq" in params else None
    except _DSLError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc

    return DslBuildStep(
        reactants=tuple((str(name), float(value)) for name, value in react.items()),
        products=tuple((str(name), float(value)) for name, value in prod.items()),
        reversible=reversible,
        forward_model=forward_model,
        kr=kr,
        standard_conc_M=C0,
        dG_eq_J_per_mol=delta_g,
        Keq_input=explicit_keq,
        line=line_number,
        line_text=line_content,
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
    try:
        parsed_step = _parse_reaction_like_step(
            stoich_part=stoich_part,
            params=params,
            energy_unit=energy_unit,
            C0=C0,
            line_number=line_number,
            line_content=line_content,
        )
    except _DSLError as exc:
        # Stoichiometry and number parsing refuse without the line; show it.
        if exc.line_number is not None:
            raise
        raise _DSLError(exc.message, line_number=line_number, line_content=line_content) from exc
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
        for statement in parameter_algebra_spec.param_statements:
            coordinate = statement.source_coordinate
            if coordinate is not None:
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


def _mechanism_build_input(ir: DSLIR):
    """Freeze the private parse accumulator into the sole builder input."""

    from kindred.core.simulator.dsl_build_input import DslMechanismBuildInput

    return DslMechanismBuildInput(
        energy_unit=ir.energy_unit,
        temperature_K=ir.temperature_K,
        standard_conc_M=ir.standard_conc_M,
        temperature_schedule=ir.temperature_schedule,
        steps=tuple(ir.steps),
    )


def prepare_dsl_request_set(
    text: str,
    *,
    units: UnitsModel | None = None,
) -> PreparedDslRequestSet:
    """Admit source once and publish its one immutable typed analysis."""

    from kindred.core.intervention_schedule_compiler import (
        compile_intervention_schedule,
    )
    from kindred.core.simulator.dsl_build import build_mechanism_from_ir
    from kindred.core.simulator.parameter_algebra_spec import (
        parse_parameter_algebra_spec_from_dsl_text,
    )
    from kindred.core.algebra.errors import AlgebraError
    from kindred.core.algebra.simulation_series import (
        reject_unknown_species_references,
    )

    units = units or UnitsModel()
    ir = _parse_dsl_ir(text, units=units)
    # Directive admission returns a nonempty typed schedule or None.
    intervention_schedule = ir.intervention_schedule
    mechanism = build_mechanism_from_ir(_mechanism_build_input(ir))
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
        reject_unknown_species_references(
            parameter_algebra_spec.observable_block,
            species_names={str(item) for item in mechanism.species_names()},
        )
    except AlgebraError as exc:
        raise _DSLError(exc.msg, line_number=exc.line, line_content=exc.line_text) from exc
    species_names = tuple(str(item) for item in mechanism.species_names())
    parameterized_intervention_schedule = (
        intervention_schedule
        if intervention_schedule is not None
        and intervention_schedule.is_parameterized
        else None
    )
    for statement in parameter_algebra_spec.observable_block.lines:
        if statement.name in species_names:
            raise _DSLError(
                f"{statement.name} is a species, so a let line cannot define it.",
                line_number=statement.line,
                line_content=statement.line_text,
            )
    if intervention_schedule is not None:
        from kindred.core.intervention_schedule import (
            InterventionScheduleError,
            ObservableTriggerCondition,
            admit_intervention_parameter_units,
            intervention_schedule_parameter_dependencies,
        )
        try:
            intervention_schedule.validate_species(species_names)
            observable_names = {statement.name for statement in parameter_algebra_spec.observable_block.lines}
            for trigger in intervention_schedule.triggers:
                if isinstance(trigger.condition, ObservableTriggerCondition) and trigger.condition.name not in observable_names:
                    raise InterventionScheduleError(
                        f"Unknown stopping observable: {trigger.condition.name}.",
                        operation="trigger", declaration_ordinal=trigger.declaration_ordinal, field="watch_observable",
                    )
            for (declaration, _nested, field), (name,) in intervention_schedule_parameter_dependencies(
                intervention_schedule
            ).items():
                if name not in parameter_algebra_spec.evaluation_dependencies:
                    raise InterventionScheduleError(
                        f"Unknown intervention parameter {name!r} in {field}_param. "
                        f"Declare it with 'param {name} = ...' or use a rate constant such as k1.",
                        declaration_ordinal=declaration, field=f"{field}_param",
                    )
            admit_intervention_parameter_units(
                intervention_schedule,
                dict(parameter_algebra_spec.quantity_units),
                dict(parameter_algebra_spec.input_units),
            )
            intervention_execution = (
                None
                if parameterized_intervention_schedule is not None
                else compile_intervention_schedule(intervention_schedule)
            )
        except InterventionScheduleError as exc:
            # Each of these refusals names its declaration, or the declarations
            # that conflict; show the line of the (latest) declaration.
            ordinals = exc.details.get("declaration_ordinals") or (exc.declaration_ordinal,)
            line_number, line_content = ir.intervention_lines[ordinals[-1]]
            raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
    else:
        intervention_execution = compile_intervention_schedule(None)
    analysis = DslAnalysis(
        origins=_source_origins(
            text,
            steps=ir.steps,
            parameter_algebra_spec=parameter_algebra_spec,
        ),
        species_names=species_names,
        observable_names=tuple(sorted(parameter_algebra_spec.observable_names)),
        inline_initial_conditions=ir.inline_initial_conditions,
        temperature_K=float(ir.temperature_K),
        standard_conc_M=float(ir.standard_conc_M),
        energy_unit=str(ir.energy_unit),
        intervention_lines=ir.intervention_lines,
    )
    return PreparedDslRequestSet(
        analysis=analysis,
        mechanism=mechanism,
        intervention_schedule=parameterized_intervention_schedule,
        intervention_execution=intervention_execution,
        parameter_algebra_spec=parameter_algebra_spec,
    )


def _analysis_source_coordinates(
    analysis: DslAnalysis,
) -> tuple[tuple[DslSourceCoordinate, DslSourceOrigin], ...]:
    """Return each admitted coordinate with its retained line owner."""

    return tuple(
        (coordinate, origin)
        for origin in analysis.origins
        for coordinate in origin.coordinates
    )


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
    names = tuple(item.name for item in updates)
    if len(names) != len(set(names)):
        raise ValueError("DSL parameter update names must be unique.")

    spec = base.parameter_algebra_spec_for_execution()
    raw_lines = source_text.splitlines(keepends=True)
    source_lines = [_line_content(line) for line in raw_lines]
    line_endings = [
        line[len(content):]
        for line, content in zip(raw_lines, source_lines)
    ]
    if not raw_lines and source_text:
        source_lines = [source_text]
        line_endings = [""]

    # Coordinates retain their physical line index and exact half-open span in
    # the source this analysis was admitted from.
    # A mechanism parameter can own two spans: its reaction-line field and
    # its param line. The admitted assignment kind selects the writable one.
    source_coordinates: dict[str, dict[str, DslSourceCoordinate]] = {}
    for coordinate, _origin in _analysis_source_coordinates(base.analysis):
        source_coordinates.setdefault(coordinate.canonical_name, {})[coordinate.role] = coordinate
    replacements: dict[int, list[tuple[int, int, str]]] = {}

    def retain_replacement(
        coordinate: DslSourceCoordinate,
        replacement: str,
    ) -> None:
        replacements.setdefault(int(coordinate.physical_line_index), []).append(
            (coordinate.value_start, coordinate.value_end, replacement)
        )

    for update in updates:
        spans = source_coordinates.get(update.name)
        if spans is None:
            raise LookupError(
                f"Parameter {update.name!r} is absent from retained DSL."
            )
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
        if assignment_kind in {"independent scalar", "mechanism input"}:
            coordinate = spans.get("param")
            if coordinate is None:
                raise ValueError(
                    f"Scalar parameter {update.name!r} is not writable."
                )
            retain_replacement(coordinate, six_digit_source_text(update.value))
            continue
        # A source identity's param line only restates its reaction field,
        # which stays the writable coordinate.
        # Callers send only enumerated writable coordinates.
        (coordinate,) = (item for role, item in spans.items() if role != "param")
        # Physical and authored indexed source coordinates are already in the
        # exact units and representation expected by their retained span.
        retain_replacement(coordinate, six_digit_source_text(update.value))

    # Parser spans are unique and never overlap.
    for line_index, line_replacements in replacements.items():
        ordered_replacements = sorted(
            line_replacements,
            key=lambda item: item[0],
            reverse=True,
        )
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
