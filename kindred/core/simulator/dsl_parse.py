"""
Computational Simulator DSL tools.

Current contract
----------------
- Steps may be bare or prefixed with `reaction:` or `equilibrium:`.
- Irreversible arrows: `->` and `=>`; reversible arrows: `<->` and `<=>`.
- Accepted keys with normalization:
  degeneracy, κ, C°|C0, dG_act|ΔG‡, Ea, dG_eq|ΔG°, k|kf|kr, A,
  energy=kcal/mol|kJ/mol|J/mol
- Packaged preset sources use this same public syntax and remain admitted by
  the same parser.
- Source-level `T=` is rejected; set nominal temperature in the mechanism tab.
- If dG_eq/ΔG° is provided and Keq is omitted, compute concentration-form
  Keq = exp(-ΔG°/(R*T)) * C0**delta_n with reaction-dependent units.
- Default per step is Eyring; Arrhenius override via A/Ea.
- Units inferred from molecularity (1/(M^(n−1)*s)); bimolecular Eyring divides by C°.

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

import copy
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import re

from ..equilibrium_rate_authority import (
    admit_explicit_directional_rate_pair,
    derived_equilibrium_ratio_from_directional_rates,
    validate_equilibrium_rate_authority_flags,
)
from ..mechanism_source import (
    MechanismAuthoringSource,
    normalize_source_for_analysis,
    source_matches_analysis,
)
from ..units import UnitsModel
from ..units import (
    admit_arrhenius_prefactor,
    admit_energy_j_per_mol,
    admit_keq,
    admit_rate,
)
from ..species import validate_c0, validate_concentration, validate_temperature
from .errors import (
    DSLError as _DSLError,
    invalid_number_error,
    missing_arrow_error,
    invalid_species_term_error,
    empty_stoichiometry_error,
    invalid_keyvalue_pair_error,
)
from .parameter_algebra_spec import classify_parameter_algebra_declaration
from .parameter_namespace import build_namespace_from_mechanism
from .dsl_build_input import DslBuildStep
from .dsl_parameter_scan import DslParameterAnalysis
from .dsl_derivation import (
    authored_step_origin,
    coordinate_source_parts,
    origin_line_positions,
    rewrite_coordinate_source_line,
    rewrite_scalar_source_line,
    scalar_origin,
)

if TYPE_CHECKING:
    from ..intervention_schedule import InterventionSchedule
    from ..intervention_schedule_compiler import InterventionExecutionContract
    from ..temperature import TemperatureScheduleProtocol
    from .wegscheider_symbolic import WegscheiderCyclicityReport

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
    "kappa": "κ",
    "κ": "κ",
    "c°": "C0",
    "c0": "C0",
    # energies / model params
    "dg_act": "dG_act",
    "δg‡": "dG_act",
    "δg^‡": "dG_act",
    "ea": "Ea",
    "dg_eq": "dG_eq",
    "δg°": "dG_eq",
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
_STOICH_TERM_SPLIT_RE = re.compile(r"(?<![eE])\+")
_COMMA_SEMI_SPLIT_RE = re.compile(r"[,;]")
_SEMI_SPLIT_RE = re.compile(r"[;]")

_REACTION_KNOWN_KEYS = frozenset(
    {"κ", "degeneracy", "kf", "k", "kr", "A", "Ea", "Keq", "dG_eq", "dG_act"}
)
_EQUILIBRIUM_KNOWN_KEYS = frozenset({"Keq", "kf", "kr", "dG_eq", "dg_eq"})


# ------------------------------ data models ----------------------------------

@dataclass
class DSLIR:
    """
    Structured intermediate representation (IR) for the simulator DSL.

    Immutable analysis and Mechanism construction are derived from this IR to
    prevent semantic drift across entry points.
    """

    version: int
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

    def __post_init__(self) -> None:
        if (
            type(self.authored_ordinal) is not int
            or self.authored_ordinal < 0
        ):
            raise ValueError("DSL origin ordinal must be a non-negative int.")
        if type(self.source_text) is not str:
            raise TypeError("DSL origin source text must be an exact string.")


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
    parameters: tuple[DslParameterAnalysis, ...]
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
        if type(self.parameters) is not tuple or any(
            type(item) is not DslParameterAnalysis
            for item in self.parameters
        ):
            raise TypeError(
                "DSL analysis parameters must be DslParameterAnalysis values."
            )
        if len({item.name for item in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("DSL analysis parameter names must be unique.")
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
        "__mechanism",
        "__intervention_schedule",
        "__intervention_execution",
        "__parameter_algebra_spec",
        "__wegscheider_admission_report",
    )

    def __init__(
        self,
        *,
        analysis: DslAnalysis,
        mechanism: object,
        intervention_schedule: InterventionSchedule | None,
        intervention_execution: InterventionExecutionContract | None,
        parameter_algebra_spec: object,
        wegscheider_admission_report: WegscheiderCyclicityReport,
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
        from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport
        if not isinstance(wegscheider_admission_report, WegscheiderCyclicityReport):
            raise TypeError("prepared DSL requires its typed Wegscheider admission report.")
        self.analysis = analysis
        self.__mechanism = mechanism
        self.__intervention_schedule = intervention_schedule
        self.__intervention_execution = intervention_execution
        self.__parameter_algebra_spec = parameter_algebra_spec
        self.__wegscheider_admission_report = wegscheider_admission_report

    def mechanism_for_execution(self) -> object:
        return copy.deepcopy(self.__mechanism)

    def parameterized_intervention_schedule_for_binding(self) -> InterventionSchedule | None:
        return copy.deepcopy(self.__intervention_schedule)

    def intervention_execution_contract(self) -> InterventionExecutionContract | None:
        return self.__intervention_execution

    def parameter_algebra_spec_for_execution(self) -> object:
        return copy.deepcopy(self.__parameter_algebra_spec)

    def wegscheider_admission_report(self) -> WegscheiderCyclicityReport:
        """Return immutable source-admission truth captured before Algebra."""
        return self.__wegscheider_admission_report
# ------------------------------ helpers --------------------------------------

def _norm_key(k: str) -> str:
    key = k.strip()
    normalized = key.casefold()
    if normalized == "keq":
        return "Keq"
    # Bare uppercase K is not lowercase k and is not accepted as a Keq alias.
    if key == "K":
        return "K"
    # Otherwise normalize to casefolded Unicode before lookup.  The table is
    # stored in the same normalized form, including Greek Delta aliases.
    return _KEY_ALIASES.get(normalized, key)


def _parse_keyvals(
    rest: str,
    *,
    reject_duplicate_canonical_keys: bool = False,
    line_number: int | None = None,
    line_content: str | None = None,
) -> Dict[str, str]:
    """
    Parse comma or semicolon-separated key=value pairs with lenient spacing.

    Example: "A=1.2e10, Ea=50, energy=kJ/mol" -> {"A":"1.2e10","Ea":"50","energy":"kJ/mol"}
    Example: "kf=1.5; kr=0.25" -> {"kf":"1.5","kr":"0.25"}
    """
    out: Dict[str, str] = {}
    original_spellings: Dict[str, str] = {}
    if not rest.strip():
        return out
    # Split on both commas and semicolons to support both formats
    for chunk in _COMMA_SEMI_SPLIT_RE.split(rest):
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
    return out


def _parse_species_side(text: str) -> Dict[str, float]:
    """
    Parse a side like "A + 2B + 0.5*C" into {name: coeff}.
    Coefficients default to 1.0 when omitted.
    """
    if not text.strip():
        return {}
    result: Dict[str, float] = {}
    for term in _STOICH_TERM_SPLIT_RE.split(text):
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


def _derive_equilibrium_rates_with_context(
    *,
    line_number: Optional[int],
    line_content: Optional[str],
    **kwargs,
):
    """
    Translate equilibrium authority errors into source-aware DSL errors.
    """
    try:
        from ..equilibrium_rate_authority import (
            derive_fast_equilibrium_rates,
        )

        if "T" in kwargs:
            kwargs["temperature_K"] = kwargs.pop("T")
        return derive_fast_equilibrium_rates(**kwargs)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
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
    if "κ" in params and family != "Eyring":
        raise _DSLError(
            "Per-step κ is valid only for an Eyring forward rate.",
            line_number=line_number,
            line_content=line_content,
        )
    return family


# ------------------------------ main parsing ---------------------------------

_DSL_IR_VERSION = 1
_NAMED_INITIAL_BLOCK_START_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{\s*(?P<body>.*?)\s*$"
)
_BRACKET_INITIAL_ASSIGNMENT_RE = re.compile(
    r"^\[(?P<species>[A-Za-z_][A-Za-z0-9_]*)\]\s*=\s*(?P<value>.+?)\s*$"
)
_INITIAL_PROVENANCE_PREFIX = "# Kindred IC provenance | "


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
            _INITIAL_PROVENANCE_PREFIX + line.rstrip()
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

    # Parse temperature schedule if present
    from ..temperature_dsl import parse_temperature_schedule
    from ..intervention_schedule import _admit_intervention_directives

    temperature_schedule = (
        parse_temperature_schedule(
            temperature_directive,
            nominal_temperature=temperature_K,
        )
        if temperature_directive is not None
        else None
    )
    intervention_schedule = _admit_intervention_directives(text)

    raw_lines = text.splitlines()
    if not any(line.strip() and not line.strip().startswith("#") for line in raw_lines):
        raise _DSLError("DSL text is empty. Provide at least one reaction or directive.")
    lines = [ln.rstrip() for ln in raw_lines]

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
        elif lower.startswith("c0=") or lower.startswith("c°="):
            key = "C0"
        if key is None:
            continue
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
        notes.append(f"C° set to {standard_conc_M:g} M")

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

        if lower.startswith("t="):
            raise _DSLError(
                "Source-level T= is not supported; the mechanism tab "
                "temperature is the nominal scientific authority.",
                line_number=line_no,
                line_content=raw,
            )

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
            steps.append(
                _parse_reaction_step(
                    rest,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        # equilibrium: line
        if lower.startswith("equilibrium:"):
            _, rest = line.split(":", 1)
            parsed_step = _parse_equilibrium_step(
                rest,
                energy_unit=energy_unit,
                T=temperature_K,
                C0=standard_conc_M,
                line_number=line_no,
                line_content=raw,
            )
            steps.append(parsed_step)
            continue

        # Fallback: Try to parse as bare arrow syntax (no "reaction:" or "equilibrium:" prefix)
        if _ARROW_RE.search(line):
            steps.append(
                _parse_bare_arrow_step(
                    line,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        raise _DSLError(f"unrecognized line: {line!r}", line_number=line_no, line_content=raw)

    if temperature_schedule is not None:
        notes.append(f"Temperature schedule detected: {temperature_schedule}")
    if intervention_schedule is not None:
        notes.append("Intervention schedule detected")

    return DSLIR(
        version=_DSL_IR_VERSION,
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
) -> Tuple[str, Dict[str, str]]:
    stoich_part, *tail = _SEMI_SPLIT_RE.split(text, maxsplit=1)
    params = (
        _parse_keyvals(
            tail[0],
            reject_duplicate_canonical_keys=reject_duplicate_canonical_keys,
            line_number=line_number,
            line_content=line_content,
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
            "Per-reaction T= is not supported; set the nominal temperature "
            "in the mechanism tab.",
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
    T: float,
    C0: float,
    missing_eyring_message: str,
    line_number: int | None,
    line_content: str | None,
) -> DslBuildStep:
    from ..kinetics import arrhenius_rate, eyring_rate
    from .common import molecularity
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
    _reject_unknown_params(params, _REACTION_KNOWN_KEYS, "reaction",
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
    kappa_raw = _float_or_none(params.get("κ"))
    if kappa_raw is not None:
        kappa = kappa_raw
    else:
        kappa = 1.0
    n = molecularity(react)

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
    if "Keq" in params:
        _Keq_early = _float_or_none(params["Keq"])
        if _Keq_early is None:
            raise _DSLError(
                "Keq must be numeric",
                line_number=line_number,
                line_content=line_content,
            )
        if _Keq_early <= 0:
            raise _DSLError(
                f"Keq must be positive, got {_Keq_early}",
                line_number=line_number, line_content=line_content,
            )
    arrhenius_A = None
    arrhenius_EaJ = None
    eyring_dGJ = None
    dG_eqJ_for_step = None
    Keq = None
    Keq_input = None

    if model == "Arrhenius":
        A = _float_or_none(params.get("A"))
        Ea = params.get("Ea")
        if A is None or Ea is None:
            raise _DSLError(
                "Arrhenius requires A and Ea",
                examples=["A -> B ; A=1e13 ; Ea=50"],
                line_number=line_number, line_content=line_content,
            )
        try:
            A = admit_arrhenius_prefactor(A, field="Arrhenius A")
        except ValueError as exc:
            raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
        Ea_val = _float_or_none(Ea)
        if Ea_val is None:
            raise _DSLError("Ea must be numeric", line_number=line_number, line_content=line_content)
        EaJ = normalize_energy_to_J_per_mol(Ea_val, energy_unit)
        try:
            EaJ = admit_energy_j_per_mol(EaJ, field="Ea")
        except ValueError as exc:
            raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
        arrhenius_A = A
        arrhenius_EaJ = EaJ
        try:
            kf = arrhenius_rate(A, EaJ, T)
        except (ValueError, OverflowError) as exc:
            raise _DSLError(
                "Arrhenius rate computation overflowed (A and Ea produce an unrepresentable rate)",
                line_number=line_number, line_content=line_content,
            ) from exc
    elif model == "Eyring":
        if kf is None:
            if "dG_act" not in params:
                raise _DSLError(
                    missing_eyring_message,
                    examples=["A -> B ; dG_act=75.5", "A -> B ; k=1.5"],
                    line_number=line_number, line_content=line_content,
                )
            dG_act_val = _float_or_none(params["dG_act"])
            if dG_act_val is None:
                raise _DSLError("dG_act must be numeric", line_number=line_number, line_content=line_content)
            dGJ = normalize_energy_to_J_per_mol(dG_act_val, energy_unit)
            try:
                dGJ = admit_energy_j_per_mol(dGJ, field="dG_act")
            except ValueError as exc:
                raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
            eyring_dGJ = dGJ
            if kappa <= 0:
                raise _DSLError(
                    f"Per-step kappa must be positive for Eyring rate computation, got {kappa}",
                    line_number=line_number, line_content=line_content,
                )
            try:
                kf = eyring_rate(
                    dGJ,
                    T,
                    kappa=kappa,
                    molecularity=n,
                    standard_conc_M=C0,
                    degeneracy=degeneracy,
                )
            except (ValueError, OverflowError) as exc:
                raise _DSLError(
                    "Eyring rate computation overflowed (dG_act too large for given T)",
                    line_number=line_number, line_content=line_content,
                ) from exc
    else:
        if kf is None:
            raise _DSLError(
                "Constant reaction requires explicit k or kf",
                line_number=line_number,
                line_content=line_content,
            )

    if reversible and kr is None:
        explicit_keq = _float_or_none(params.get("Keq"))
        if "dG_eq" in params:
            dG_eq_value = _float_or_none(params["dG_eq"])
            if dG_eq_value is None:
                raise _DSLError(
                    "dG_eq must be numeric",
                    line_number=line_number,
                    line_content=line_content,
                )
            dG_eqJ_for_step = normalize_energy_to_J_per_mol(
                dG_eq_value,
                energy_unit,
            )
            try:
                dG_eqJ_for_step = admit_energy_j_per_mol(
                    dG_eqJ_for_step,
                    field="dG_eq",
                )
            except ValueError as exc:
                raise _DSLError(
                    str(exc),
                    line_number=line_number,
                    line_content=line_content,
                ) from exc
        rates = _derive_equilibrium_rates_with_context(
            Keq=explicit_keq,
            dG_eq_J_per_mol=dG_eqJ_for_step,
            T=T,
            standard_conc_M=C0,
            delta_n=math.fsum(prod.values()) - math.fsum(react.values()),
            explicit_rates=(float(kf),),
            line_number=line_number,
            line_content=line_content,
        )
        kf = rates.kf
        kr = rates.kr
        Keq = rates.Keq
        Keq_input = explicit_keq

    return DslBuildStep(
        reactants=tuple(
            (str(name), float(value)) for name, value in react.items()
        ),
        products=tuple(
            (str(name), float(value)) for name, value in prod.items()
        ),
        reversible=reversible,
        kf=float(kf),
        kr=float(kr) if kr is not None else None,
        model=model,
        is_equilibrium=False,
        arrhenius_A=arrhenius_A,
        arrhenius_Ea_J_per_mol=arrhenius_EaJ,
        eyring_dG_act_J_per_mol=eyring_dGJ,
        kappa=kappa,
        standard_conc_M=C0,
        dG_eq_J_per_mol=dG_eqJ_for_step,
        Keq=float(Keq) if Keq is not None else None,
        Keq_input=Keq_input,
        user_kf_explicit=any(
            key in params for key in ("kf", "k", "A", "Ea", "dG_act")
        ),
        user_kr_explicit=bool(params.get("kr")),
        degeneracy=degeneracy,
    )


def _parse_reaction_step(
    rest: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    line_number: int,
    line_content: str,
) -> DslBuildStep:
    stoich_part, params = _split_stoich_and_params(
        rest,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
    )
    return _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        T=T,
        C0=C0,
        missing_eyring_message="Eyring step requires dG_act or explicit k",
        line_number=line_number,
        line_content=line_content,
    )


def _parse_bare_arrow_step(
    line: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    line_number: int,
    line_content: str,
) -> DslBuildStep:
    stoich_part, params = _split_stoich_and_params(
        line,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
    )
    return _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        T=T,
        C0=C0,
        missing_eyring_message="Bare arrow syntax requires k (or kf), or dG_act for Eyring",
        line_number=line_number,
        line_content=line_content,
    )


def _parse_equilibrium_step(
    rest: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    line_number: int,
    line_content: str,
) -> DslBuildStep:
    from .common import molecularity
    from .kinetics import normalize_energy_to_J_per_mol

    stoich_part, params = _split_stoich_and_params(
        rest,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
    )
    try:
        react, prod, arrow = _parse_stoich(stoich_part)
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
    _ = molecularity(react)

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _EQUILIBRIUM_KNOWN_KEYS, "equilibrium",
                           line_number=line_number, line_content=line_content)

    has_explicit_Keq = "Keq" in params
    has_explicit_dG_eq = "dG_eq" in params
    Keq = _float_or_none(params.get("Keq"))
    if has_explicit_Keq and Keq is None:
        raise _DSLError(
            "Keq must be numeric",
            line_number=line_number,
            line_content=line_content,
        )
    if has_explicit_Keq and Keq is not None:
        _validate_rate_or_K(Keq, "Keq", line_number=line_number, line_content=line_content)
    dG_eqJ = None

    kf_explicit = None
    kr_explicit = None
    if "kf" in params:
        kfv = _float_or_none(params["kf"])
        if kfv is None:
            raise _DSLError("kf must be numeric", line_number=line_number, line_content=line_content)
        _validate_rate_or_K(kfv, "kf", line_number=line_number, line_content=line_content)
        kf_explicit = kfv
    if "kr" in params:
        krv = _float_or_none(params["kr"])
        if krv is None:
            raise _DSLError("kr must be numeric", line_number=line_number, line_content=line_content)
        _validate_rate_or_K(krv, "kr", line_number=line_number, line_content=line_content)
        kr_explicit = krv

    _validate_equilibrium_rate_authority_for_dsl(
        has_kf=kf_explicit is not None,
        has_kr=kr_explicit is not None,
        has_Keq=has_explicit_Keq,
        has_dG_eq=has_explicit_dG_eq,
        line_number=line_number,
        line_content=line_content,
    )

    explicit_directional_pair = (
        not has_explicit_Keq
        and not has_explicit_dG_eq
        and kf_explicit is not None
        and kr_explicit is not None
    )
    if explicit_directional_pair:
        try:
            kf_explicit, kr_explicit = admit_explicit_directional_rate_pair(
                kf_explicit,
                kr_explicit,
            )
            _ = derived_equilibrium_ratio_from_directional_rates(
                kf_explicit,
                kr_explicit,
            )
        except ValueError as exc:
            raise _DSLError(
                str(exc),
                line_number=line_number,
                line_content=line_content,
            ) from exc

    if Keq is None and has_explicit_dG_eq:
        dG_eq = (
            params["dG_eq"]
            if "dG_eq" in params
            else params.get("dg_eq")
        )
        dG_eq_val = _float_or_none(dG_eq)
        if dG_eq_val is None:
            raise _DSLError("dG_eq must be numeric", line_number=line_number, line_content=line_content)
        dG_eqJ = normalize_energy_to_J_per_mol(dG_eq_val, energy_unit)
        try:
            dG_eqJ = admit_energy_j_per_mol(dG_eqJ, field="dG_eq")
        except ValueError as exc:
            raise _DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
    if explicit_directional_pair:
        if kf_explicit is None or kr_explicit is None:
            raise AssertionError("explicit directional admission lost its pair")
        kf = kf_explicit
        kr = kr_explicit
        derived_Keq = None
    else:
        fe = _derive_equilibrium_rates_with_context(
            Keq=Keq,
            dG_eq_J_per_mol=dG_eqJ,
            T=T,
            standard_conc_M=C0,
            delta_n=math.fsum(prod.values()) - math.fsum(react.values()),
            explicit_rates=(kf_explicit,),
            line_number=line_number,
            line_content=line_content,
        )
        kf = fe.kf if kf_explicit is None else kf_explicit
        kr = fe.kr if kr_explicit is None else kr_explicit
        derived_Keq = float(fe.Keq)

    return DslBuildStep(
        reactants=tuple(
            (str(name), float(value)) for name, value in react.items()
        ),
        products=tuple(
            (str(name), float(value)) for name, value in prod.items()
        ),
        reversible=True,
        kf=float(kf),
        kr=float(kr) if kr is not None else None,
        model="Eyring",
        is_equilibrium=True,
        eyring_dG_act_J_per_mol=None,
        kappa=1.0,
        standard_conc_M=C0,
        dG_eq_J_per_mol=dG_eqJ,
        Keq=derived_Keq,
        Keq_input=(Keq if has_explicit_Keq else None),
        user_kf_explicit=("kf" in params),
        user_kr_explicit=("kr" in params),
        degeneracy=1,
    )


def _source_origins(text: str) -> tuple[DslSourceOrigin, ...]:
    origins: list[DslSourceOrigin] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        origins.append(
            DslSourceOrigin(
                authored_ordinal=len(origins),
                source_text=raw_line,
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
    from kindred.core.temperature import TemperatureSchedule

    if type(ir) is not DSLIR:
        raise TypeError("DSL builder freeze requires the exact private DSLIR.")
    canonical_initials = {
        str(name): float(value)
        for name, value in dict(initials or {}).items()
    }

    temperature_schedule = ir.temperature_schedule
    if ir.temperature_schedule is not None:
        if type(ir.temperature_schedule) is not TemperatureSchedule:
            raise TypeError(
                "DSL builder freeze requires exact TemperatureSchedule."
            )

    return DslMechanismBuildInput(
        version=1,
        energy_unit=str(ir.energy_unit),
        temperature_K=float(ir.temperature_K),
        standard_conc_M=float(ir.standard_conc_M),
        temperature_schedule=temperature_schedule,
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
    from kindred.core.simulator.dsl_parameter_scan import (
        extract_parameters_from_mechanism,
    )
    from kindred.core.simulator.parameter_algebra_spec import (
        parse_parameter_algebra_spec_from_dsl_text,
    )
    from kindred.core.algebra.simulation_series import (
        compile_algebra_observables,
    )

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
    from kindred.core.simulator.wegscheider_symbolic import (
        analyze_wegscheider_cyclicity,
    )

    # Capture admission from canonical parsed coefficient/equilibrium authority
    # before Algebra compilation, request overrides, bindings, or runtime state
    # can mutate the executable mechanism.
    wegscheider_admission_report = analyze_wegscheider_cyclicity(mechanism)
    mechanism_namespace = build_namespace_from_mechanism(mechanism)
    parameter_algebra_spec = parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_namespace=mechanism_namespace,
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
    definitions = tuple(extract_parameters_from_mechanism(mechanism))
    species_names = tuple(str(item) for item in mechanism.species_names())
    if intervention_schedule is not None:
        intervention_schedule.validate_species(species_names)
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
        origins=_source_origins(text),
        species_names=species_names,
        observable_names=tuple(
            sorted(
                str(name)
                for name in parameter_algebra_spec.observable_names
                if str(name).strip()
            )
        ),
        parameters=definitions,
        inline_initial_conditions=ir.inline_initial_conditions,
        temperature_K=float(ir.temperature_K),
        energy_unit=str(ir.energy_unit),
    )
    metadata = getattr(mechanism, "metadata", None)
    if isinstance(metadata, dict):
        metadata["parameter_algebra_spec"] = parameter_algebra_spec
    return PreparedDslRequestSet(
        analysis=analysis,
        mechanism=mechanism,
        intervention_schedule=parameterized_intervention_schedule,
        intervention_execution=intervention_execution,
        parameter_algebra_spec=parameter_algebra_spec,
        wegscheider_admission_report=wegscheider_admission_report,
    )


def _forward_model_payload(model: object) -> dict[str, object]:
    from kindred.core.kinetics import (
        ArrheniusRateModel,
        ConstantRateModel,
        EyringRateModel,
    )

    if isinstance(model, ConstantRateModel):
        return {"type": "Constant", "k": float(model.k)}
    if isinstance(model, ArrheniusRateModel):
        return {
            "type": "Arrhenius",
            "A": float(model.A),
            "Ea_J_per_mol": float(model.Ea_J_per_mol),
        }
    if isinstance(model, EyringRateModel):
        return {
            "type": "Eyring",
            "dG_act_J_per_mol": float(model.dG_act_J_per_mol),
            "degeneracy": int(model.degeneracy),
            "kappa": float(model.kappa),
            "standard_conc_M": float(model.standard_conc_M),
        }
    raise TypeError("Coordinate update lost its typed rate family.")


def _reaction_model_overrides(model: object) -> dict[str, object]:
    payload = _forward_model_payload(model)
    family = str(payload.pop("type"))
    payload.pop("k", None)
    return {"model": family, **payload}


def derive_source_with_parameter_updates(
    source_text: str,
    base: PreparedDslRequestSet,
    updates: tuple[DslParameterCoordinateUpdate, ...],
    *,
    wegscheider_cyclicity_enabled: bool,
) -> tuple[str, PreparedDslRequestSet]:
    """Apply typed updates to exact source and return both authorities."""

    from kindred.core.equilibrium_rate_authority import (
        EquilibriumRateAuthorityKind,
        effective_equilibrium_keq,
        effective_reverse_rate_from_keq,
        normalize_existing_equilibrium_rate_authority,
    )
    from kindred.core.kinetics import (
        ArrheniusRateModel,
        EyringRateModel,
        forward_rate_at_temperature,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
        write_reference_rate,
    )
    from kindred.core.mechanism_metadata import (
        EquilibriumMetadataKeys,
        MechanismMetadataKeys,
        MechanismMetadataView,
    )
    from kindred.core.simulator.step_indexing import (
        lookup_step_param_target,
    )

    if type(base) is not PreparedDslRequestSet:
        raise TypeError("base must be exact PreparedDslRequestSet.")
    if type(source_text) is not str:
        raise TypeError("source_text must be an exact str.")
    if base.analysis.inline_initial_conditions.has_declarations:
        raise ValueError(
            "Typed parameter derivation requires canonical materialized "
            "initial conditions."
        )
    if type(updates) is not tuple or any(
        type(item) is not DslParameterCoordinateUpdate
        for item in updates
    ):
        raise TypeError(
            "updates must be an exact tuple of "
            "DslParameterCoordinateUpdate values."
        )
    if type(wegscheider_cyclicity_enabled) is not bool:
        raise TypeError(
            "wegscheider_cyclicity_enabled must be an exact bool."
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

    mechanism = base.mechanism_for_execution()
    initials = {
        str(name): float(species.initial_conc)
        for name, species in getattr(mechanism, "species", {}).items()
    }
    spec = base.parameter_algebra_spec_for_execution()
    metadata = getattr(mechanism, "metadata", None)
    if not isinstance(metadata, dict):
        raise TypeError("Prepared DSL mechanism metadata must be mutable.")
    metadata[
        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED
    ] = wegscheider_cyclicity_enabled
    metadata_view = MechanismMetadataView.from_metadata(metadata)
    temperature = float(base.analysis.temperature_K)
    if temperature != metadata_view.temperature_K:
        raise ValueError(
            "Prepared DSL nominal temperature metadata is inconsistent."
        )

    parameter_by_name = {
        str(item.name): item
        for item in base.analysis.parameters
    }
    constrained = {
        str(statement.name)
        for statement in spec.param_statements
        if lookup_step_param_target(mechanism, str(statement.name)) is not None
    }
    authored_source = source_text
    raw_lines = authored_source.splitlines(keepends=True)
    line_contents = [line.rstrip("\r\n") for line in raw_lines]
    line_endings = [
        line[len(content):]
        for line, content in zip(raw_lines, line_contents)
    ]
    if not raw_lines and authored_source:
        line_contents = [authored_source]
        line_endings = [""]
    source_lines = line_contents
    origin_positions = origin_line_positions(base.analysis, authored_source)
    updated_origin_lines: dict[int, str] = {}
    scalar_values = metadata.get("scalar_params")
    if not isinstance(scalar_values, dict):
        scalar_values = {}
        metadata["scalar_params"] = scalar_values

    for update in updates:
        target = lookup_step_param_target(mechanism, update.name)
        definition = parameter_by_name.get(update.name)
        scalar_statement_known = any(
            statement.name == update.name
            for statement in spec.param_statements
        )
        if target is None and not scalar_statement_known:
            raise LookupError(
                f"Parameter {update.name!r} is absent from retained DSL."
            )
        if (
            target is not None
            and (
                definition is None
                or not bool(definition.editable)
            )
        ) or update.name in constrained:
            raise ValueError(
                f"Parameter {update.name!r} is not writable."
            )
        if target is None:
            scalar_info = dict(
                dict(metadata.get("scalar_param_info") or {}).get(
                    update.name
                )
                or {}
            )
            if (
                scalar_info.get("derived") is True
                or scalar_info.get("editable") is False
            ):
                raise ValueError(
                    f"Scalar parameter {update.name!r} is not writable."
                )
            statements = list(spec.param_statements)
            statement_index = next(
                (
                    index
                    for index, statement in enumerate(statements)
                    if statement.name == update.name
                ),
                None,
            )
            if statement_index is None:
                raise LookupError(
                    f"Scalar parameter {update.name!r} has no typed declaration."
                )
            origin = scalar_origin(
                base.analysis,
                name=update.name,
            )
            old_line = updated_origin_lines.get(
                origin.authored_ordinal,
                source_lines[origin_positions[origin.authored_ordinal]],
            )
            new_line = rewrite_scalar_source_line(
                old_line,
                name=update.name,
                expression=repr(float(update.value)),
            )
            updated_origin_lines[origin.authored_ordinal] = new_line
            statement = statements[statement_index]
            from kindred.core.algebra.parser import AlgebraBlock, LetStatement, NumberNode

            replacement_expr = NumberNode(float(update.value))
            statements[statement_index] = replace(
                statement,
                expr_src=repr(float(update.value)),
                line_content=new_line,
                expr_ast=replacement_expr,
            )
            retained_parameter_block = getattr(spec, "parameter_block", None)
            if not isinstance(retained_parameter_block, AlgebraBlock):
                raise ValueError(
                    f"Scalar parameter {update.name!r} has no retained parameter block."
                )
            retained_lines = tuple(
                LetStatement(
                    name=retained_statement.name,
                    expr=replacement_expr,
                    line=retained_statement.line,
                    col=retained_statement.col,
                    line_text=new_line,
                )
                if retained_statement.name == update.name
                else retained_statement
                for retained_statement in retained_parameter_block.lines
            )
            retained_static_values = retained_parameter_block.static_value_map()
            retained_static_values[update.name] = float(update.value)
            spec = replace(
                spec,
                param_statements=statements,
                parameter_block=AlgebraBlock(
                    lines=retained_lines,
                    ast=tuple(item.expr for item in retained_lines),
                    static_values=retained_static_values,
                ),
            )
            scalar_values[update.name] = update.value
            continue

        kind, index, role, entry = target
        origin = authored_step_origin(
            base.analysis,
            step_index=int(entry["step_index"]),
        )
        old_line = updated_origin_lines.get(
            origin.authored_ordinal,
            source_lines[origin_positions[origin.authored_ordinal]],
        )
        _prefix, tokens, _comment = coordinate_source_parts(old_line)
        source_token = role
        source_value = update.value

        if kind == "reaction":
            reaction = mechanism.reactions[index]
            model = forward_rate_model_for_reaction(
                reaction,
                default_standard_conc_M=metadata_view.standard_conc_M,
            )
            rewritten = write_reference_rate(
                model,
                reference_rate=update.value,
                reference_temperature_K=temperature,
            )
            rate = forward_rate_at_temperature(rewritten, temperature)
            mechanism.reactions[index] = replace(
                reaction,
                rate=rate,
                overrides=_reaction_model_overrides(rewritten),
            )
            if isinstance(rewritten, ArrheniusRateModel):
                source_token = "Ea"
                source_value = UnitsModel(
                    energy_unit=metadata_view.energy_unit,
                ).from_jmol(rewritten.Ea_J_per_mol)
            elif isinstance(rewritten, EyringRateModel):
                source_token = "dG_act"
                source_value = UnitsModel(
                    energy_unit=metadata_view.energy_unit,
                ).from_jmol(rewritten.dG_act_J_per_mol)
            else:
                authored_forward_tokens = [
                    pair[0]
                    for pair in tokens
                    if str(pair[0]).strip().casefold() in {"k", "kf"}
                ]
                if len(authored_forward_tokens) != 1:
                    raise ValueError(
                        f"Parameter {update.name!r} has no unique admitted "
                        "irreversible forward-rate token."
                    )
                source_token = authored_forward_tokens[0]
        elif kind == "equilibrium":
            equilibrium = mechanism.equilibria[index]
            equilibrium_metadata = dict(equilibrium.metadata)
            authority = normalize_existing_equilibrium_rate_authority(
                equilibrium
            )
            if role == "kf":
                model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=temperature,
                    default_standard_conc_M=metadata_view.standard_conc_M,
                )
                rewritten = write_reference_rate(
                    model,
                    reference_rate=update.value,
                    reference_temperature_K=temperature,
                )
                kf = forward_rate_at_temperature(rewritten, temperature)
                equilibrium_metadata[
                    EquilibriumMetadataKeys.FORWARD_MODEL
                ] = _forward_model_payload(rewritten)
                kr = equilibrium.kr
                if authority.kind == EquilibriumRateAuthorityKind.KEQ:
                    keq = effective_equilibrium_keq(
                        equilibrium,
                        temperature_K=temperature,
                    )
                    if keq is None:
                        raise ValueError(
                            f"Parameter {update.name!r} lost Keq authority."
                        )
                    kr = effective_reverse_rate_from_keq(
                        kf,
                        keq,
                    )
                mechanism.equilibria[index] = replace(
                    equilibrium,
                    kf=kf,
                    kr=kr,
                    metadata=equilibrium_metadata,
                )
                source_energy_unit = metadata_view.energy_unit
                if isinstance(rewritten, ArrheniusRateModel):
                    source_token = "Ea"
                    source_value = UnitsModel(
                        energy_unit=source_energy_unit,
                    ).from_jmol(rewritten.Ea_J_per_mol)
                elif isinstance(rewritten, EyringRateModel):
                    source_token = "dG_act"
                    source_value = UnitsModel(
                        energy_unit=source_energy_unit,
                    ).from_jmol(rewritten.dG_act_J_per_mol)
                else:
                    authored_forward_tokens = [
                        pair[0]
                        for pair in tokens
                        if str(pair[0]).strip().casefold() in {"k", "kf"}
                    ]
                    if len(authored_forward_tokens) != 1:
                        raise ValueError(
                            f"Parameter {update.name!r} has no unique admitted "
                            "reversible forward-rate token."
                        )
                    source_token = authored_forward_tokens[0]
            elif role == "kr":
                if authority.kind != EquilibriumRateAuthorityKind.KR:
                    raise ValueError(
                        f"Parameter {update.name!r} is derived, not writable."
                    )
                mechanism.equilibria[index] = replace(
                    equilibrium,
                    kr=update.value,
                    metadata=equilibrium_metadata,
                )
                source_token = "kr"
            elif role == "Keq":
                if not authority.has_explicit_keq_param:
                    raise ValueError(
                        f"Parameter {update.name!r} is derived, not writable."
                    )
                if update.value <= 0.0:
                    raise ValueError(
                        f"Parameter {update.name!r} must be positive."
                    )
                kf = float(equilibrium.kf)
                equilibrium_metadata[
                    EquilibriumMetadataKeys.KEQ_INPUT
                ] = update.value
                mechanism.equilibria[index] = replace(
                    equilibrium,
                    Keq=update.value,
                    kr=effective_reverse_rate_from_keq(
                        kf,
                        update.value,
                    ),
                    metadata=equilibrium_metadata,
                )
                source_token = "Keq"
            else:
                raise ValueError(
                    f"Unsupported equilibrium coordinate role {role!r}."
                )
        else:
            raise ValueError(
                f"Unsupported mechanism coordinate kind {kind!r}."
            )

        new_line = rewrite_coordinate_source_line(
            old_line,
            token=source_token,
            value=float(source_value),
        )
        updated_origin_lines[origin.authored_ordinal] = new_line

    for ordinal, line in updated_origin_lines.items():
        source_lines[origin_positions[ordinal]] = line
    updated_text = "".join(
        content + ending
        for content, ending in zip(source_lines, line_endings)
    )
    # Publish only the typed state admitted from the exact rendered source.
    # The in-loop mechanism is deliberately not returned: its calculated
    # values may retain precision that the source token cannot represent.
    prepared = prepare_dsl_request_set(
        updated_text,
        initials=initials,
        units=UnitsModel(
            energy_unit=base.analysis.energy_unit,
            temperature_K=base.analysis.temperature_K,
            standard_conc_M=metadata_view.standard_conc_M,
        ),
    )
    admitted_mechanism = prepared.mechanism_for_execution()
    admitted_metadata = getattr(admitted_mechanism, "metadata", None)
    if isinstance(admitted_metadata, dict):
        admitted_metadata[
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED
        ] = wegscheider_cyclicity_enabled
    result = PreparedDslRequestSet(
        analysis=prepared.analysis,
        mechanism=admitted_mechanism,
        intervention_schedule=(
            prepared.parameterized_intervention_schedule_for_binding()
        ),
        intervention_execution=prepared.intervention_execution_contract(),
        parameter_algebra_spec=prepared.parameter_algebra_spec_for_execution(),
        wegscheider_admission_report=prepared.wegscheider_admission_report(),
    )
    return updated_text, result
