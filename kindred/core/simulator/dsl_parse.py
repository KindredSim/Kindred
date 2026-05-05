"""
Computational Simulator DSL tools.

Current contract
----------------
- Entries: `reaction:`, `equilibrium:` and a `States/TS` block.
- Accepted keys with normalization:
  T, degeneracy, κ, C°|C0, dG_act|ΔG‡, Ea, dG_eq|ΔG°, k|kf|kr, A, k_fast,
  energy=kcal/mol|kJ/mol|J/mol, state, edge
- If dG_eq/ΔG° is provided and Keq is omitted, compute Keq = exp(-ΔG°/(R*T)).
- Default per step is Eyring; Arrhenius override via A/Ea.
- Units inferred from molecularity (1/(M^(n−1)*s)); bimolecular Eyring divides by C°.
- TS degree fixed to 2 and enforced; attempts to violate raise a structured error.
- Preview output uses scientific notation, 3 significant figures, and ROUND_HALF_UP:
  "A <-> B ; kf=1.23e+05 ; kr=4.56e+03  # model=Eyring; unit=1/s; κ=1.0; T=298.15 K; source=explicit"

Scope
-----
- Parse a simple line-oriented DSL (no nested syntax).
- Provide structured results and preview strings for the GUI Simulator.
- No file access; caller passes text.

Non-goals
---------
- Persisting anything, touching mechanisms directly, or solver integration.

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast, TYPE_CHECKING

import re

from ..units import UnitsModel
from .algebra_section import (
    is_bare_assignment_algebra_line,
    is_let_algebra_line,
    is_param_algebra_line,
)
from .dsl_types import StepPreview
from .errors import (
    DSLError,
    invalid_number_error,
    missing_arrow_error,
    invalid_species_term_error,
    empty_stoichiometry_error,
    invalid_keyvalue_pair_error,
    invalid_boolean_error,
)

if TYPE_CHECKING:
    from ..mechanism import Mechanism
    from ..intervention_schedule import InterventionSchedule
    from ..temperature import TemperatureScheduleProtocol
    from .dsl_parameter_scan import ParameterDefinition
    from .state_model import StateNetwork

logger = logging.getLogger(__name__)


__all__ = [
    "DSLError",
    "StepPreview",
    "DSLResult",
    "ParameterDefinition",
    "parse_and_preview",
    "parse_dsl",
    "parse_dsl_to_mechanism",
    "extract_parameters_from_dsl",
    "extract_parameter_names_from_dsl",
]


# ------------------------------ normalized keys ------------------------------

_KEY_ALIASES: Dict[str, str] = {
    # temperature / states
    "t": "T",
    "degeneracy": "degeneracy",
    "kappa": "κ",
    "κ": "κ",
    "c°": "C0",
    "c0": "C0",
    # energies / model params
    "dg_act": "dG_act",
    "Δg‡": "dG_act",
    "Δg^‡": "dG_act",
    "dG_act": "dG_act",
    "ΔG‡": "dG_act",
    "ea": "Ea",
    "Ea": "Ea",
    "dg_eq": "dG_eq",
    "Δg°": "dG_eq",
    "dG_eq": "dG_eq",
    "ΔG°": "dG_eq",
    "a": "A",
    "A": "A",
    "k": "k",
    "K": "Keq",  # Equilibrium constant (surface alias; distinct from lowercase k)
    "keq": "Keq",
    "k_eq": "Keq",
    "kf": "kf",
    "kr": "kr",
    "k_fast": "k_fast",
    # units selector
    "energy": "energy",
    # state graph
    "state": "state",
    "edge": "edge",
}

_ARROW_RE = re.compile(r"<->|<=>|->|=>")
_SPECIES_TERM_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*\*?\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*$")
_COMMA_SEMI_SPLIT_RE = re.compile(r"[,;]")
_STATE_REST_SPLIT_RE = re.compile(r"[;,]")
_SEMI_SPLIT_RE = re.compile(r"[;]")

_REACTION_KNOWN_KEYS = frozenset({"κ", "kf", "k", "kr", "A", "Ea", "Keq", "dG_eq", "dG_act"})
_EQUILIBRIUM_KNOWN_KEYS = frozenset({"Keq", "kf", "kr", "dG_eq", "dg_eq", "cm_id"})


# ------------------------------ data models ----------------------------------

@dataclass
class DSLResult:
    """
    Container for parsed simulator content.

    Fields
    ------
    previews : list[StepPreview]
        Human-readable preview lines.
    notes : list[str]
        Non-fatal normalization notes for UI display.
    temperature_schedule : TemperatureScheduleProtocol | None
        Temperature schedule for time-dependent temperature.
        None if no temperature schedule specified.
    intervention_schedule : InterventionSchedule | None
        Fixed species intervention schedule parsed from intervention directives.
    ir : DSLIR | None
        Structured intermediate representation (single source of truth for parsing).
    """
    previews: List[StepPreview] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    temperature_schedule: Optional["TemperatureScheduleProtocol"] = None
    intervention_schedule: Optional["InterventionSchedule"] = None
    ir: Optional["DSLIR"] = None


@dataclass
class DSLIR:
    """
    Structured intermediate representation (IR) for the simulator DSL.

    Preview generation and Mechanism construction are derived from this IR to
    prevent semantic drift across entry points.
    """

    version: int
    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    kappa_global: float
    temperature_schedule: Optional["TemperatureScheduleProtocol"]
    intervention_schedule: Optional["InterventionSchedule"]
    state_network: "StateNetwork"
    steps: List["ParsedStep"]
    algebra_lines: List[str]
    initials_from_dsl: Dict[str, float]
    notes: List[str] = field(default_factory=list)


# ------------------------------ helpers --------------------------------------

def _norm_key(k: str) -> str:
    key = k.strip()
    # Preserve uppercase K as the equilibrium alias distinct from lowercase k.
    if key == "K":
        return "Keq"
    # Otherwise normalize to lowercase and look up
    return _KEY_ALIASES.get(key.lower(), key)


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
            raise DSLError(
                f"Duplicate parameter: '{previous_spelling}' and '{raw_key}' both resolve to {canonical_key}",
                line_number=line_number,
                line_content=line_content,
            )
        original_spellings[canonical_key] = raw_key
        out[canonical_key] = v.strip()
    return out


_MEMBERS_TERM_RE = re.compile(r"^(\d+)?([A-Za-z_][A-Za-z0-9_]*)$")


def _parse_members_expr(expr: str) -> Tuple[str, ...]:
    """
    Parse a state `members=` expression like "A+B" or "2A+B" into a tuple with duplicates.

    This is used to recover stoichiometry for state-network conversion.
    """
    s = str(expr or "").strip()
    if not s:
        raise ValueError("members cannot be empty")
    parts = [p.strip() for p in s.split("+") if p.strip()]
    if not parts:
        raise ValueError("members cannot be empty")
    out: List[str] = []
    for raw in parts:
        term = raw.replace(" ", "")
        m = _MEMBERS_TERM_RE.match(term)
        if not m:
            raise ValueError(f"invalid members term {raw!r}")
        coeff = int(m.group(1) or "1")
        if coeff <= 0:
            raise ValueError("members coefficients must be positive integers")
        name = m.group(2)
        out.extend([name] * coeff)
    return tuple(out)


def _parse_state_rest(rest: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Parse the `state:` line body and return (name, kv).

    Supports both:
      - "name=A; kind=GS; energy=0.0"
      - "A, kind=GS, energy=0.0"
    """
    parts = [p.strip() for p in _STATE_REST_SPLIT_RE.split(str(rest or "")) if p.strip()]
    name: Optional[str] = None
    kv_parts: List[str] = []
    for part in parts:
        if "=" in part:
            kv_parts.append(part)
        elif name is None:
            name = part
    kv = _parse_keyvals(",".join(kv_parts)) if kv_parts else {}
    if not name:
        name = kv.get("state") or kv.get("name")
    return name, kv


def _extract_numeric_value(expr: str) -> Optional[float]:
    """
    Best-effort extraction of a numeric literal from a DSL value string.

    Accepts values like "1.0", "1e-3", or "12.5 kJ/mol" (unit is ignored).
    Returns None if the value is not purely numeric.
    """
    if expr is None:
        return None
    value = expr.strip()
    if not value:
        return None
    parts = value.split()
    candidate = parts[0]
    return _float_or_none(candidate)


def extract_parameters_from_dsl(text: str) -> List["ParameterDefinition"]:
    from .dsl_parameter_scan import extract_parameters_from_dsl as _impl

    return _impl(text)


def extract_parameter_names_from_dsl(text: str) -> set[str]:
    from .dsl_parameter_scan import extract_parameter_names_from_dsl as _impl

    return _impl(text)


def __getattr__(name: str):
    if name == "ParameterDefinition":
        from .dsl_parameter_scan import ParameterDefinition

        return ParameterDefinition
    raise AttributeError(name)


def _parse_species_side(text: str) -> Dict[str, float]:
    """
    Parse a side like "A + 2B + 0.5*C" into {name: coeff}.
    Coefficients default to 1.0 when omitted.
    """
    if not text.strip():
        return {}
    result: Dict[str, float] = {}
    for term in text.split("+"):
        term = term.strip()
        if not term:
            raise DSLError("Malformed stoichiometry: empty term (e.g., consecutive '+' signs)")
        m = _SPECIES_TERM_RE.match(term)
        if not m and "*" in term:
            # Allow patterns like "2 * B" while rejecting "A B"
            m = _SPECIES_TERM_RE.match(term.replace(" ", ""))
        if not m:
            raise invalid_species_term_error(term)
        coef_s, name = m.groups()
        coef = _float_or_none(coef_s) if coef_s is not None else 1.0
        if coef == 0.0:
            continue
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
    except Exception:
        logger.debug(f"Failed to convert '{s}' to float", exc_info=True)
        raise invalid_number_error(str(s), "value")
    if not math.isfinite(v):
        raise invalid_number_error(str(s), "value")
    return v


def _validate_rate_or_K(
    value: Optional[float],
    name: str,
    *,
    line_number: int | None = None,
    line_content: str | None = None,
) -> None:
    if value is None:
        return
    if math.isnan(value) or math.isinf(value) or value < 0:
        raise DSLError(
            f"'{name}' must be a non-negative finite number, got {value}",
            line_number=line_number,
            line_content=line_content,
        )


def _bool_from_str(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise invalid_boolean_error(s, "boolean")


def _derive_equilibrium_rates_with_context(
    *,
    line_number: Optional[int],
    line_content: Optional[str],
    **kwargs,
):
    """
    Call derive_equilibrium_rates while translating ValueError into DSLError with context.
    """
    try:
        from .common import derive_equilibrium_rates

        return derive_equilibrium_rates(**kwargs)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc


# ------------------------------ main parsing ---------------------------------

_DSL_IR_VERSION = 1


def _parse_dsl_ir(text: str, *, units: UnitsModel | None = None) -> "DSLIR":
    """
    Parse simulator DSL into a structured intermediate representation (IR).

    This is the single source of truth for top-level DSL parsing. Preview generation
    (`parse_dsl`) and Mechanism construction (`parse_dsl_to_mechanism`) derive their
    behavior from this IR to prevent semantic drift across entry points.
    """
    u = units or UnitsModel()
    energy_unit = u.energy_unit
    temperature_K = float(u.temperature_K)
    standard_conc_M = float(u.standard_conc_M)
    kappa_global = 1.0

    from .state_model import StateNetwork, TSDegreeError

    state_network = StateNetwork()
    steps: List[ParsedStep] = []
    algebra_lines: List[str] = []
    initials_from_dsl: Dict[str, float] = {}
    notes: List[str] = []

    # Parse temperature schedule if present
    from ..temperature_dsl import parse_temperature_schedule
    from ..intervention_schedule import parse_intervention_schedule_from_dsl

    temperature_schedule = parse_temperature_schedule(text)
    intervention_schedule = parse_intervention_schedule_from_dsl(text)

    raw_lines = text.splitlines()
    if not any(line.strip() and not line.strip().startswith("#") for line in raw_lines):
        raise DSLError("DSL text is empty. Provide at least one reaction or directive.")

    lines = [ln.rstrip() for ln in raw_lines]
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        lower = line.lower()

        if not line or line.startswith("#"):
            continue

        # Computational Mode source-of-truth lines are embedded in the Reaction DSL
        # but are not part of the simulation DSL. Simulations consume only the
        # generated mechanism block, so `comp:` lines must be ignored here.
        if lower.startswith("comp:"):
            continue

        # Temperature schedule lines: parsed separately above
        if lower.startswith(("time:", "temp_const:", "temp_step:", "temp_response:")):
            continue

        if lower.startswith("intervention:"):
            continue

        # Header-like switches
        if lower.startswith("energy="):
            try:
                energy_unit = _parse_energy_unit_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"energy unit set to {energy_unit}")
            continue

        if lower.startswith("t="):
            try:
                temperature_K = _parse_temperature_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"T set to {temperature_K:.2f} K")
            continue

        if lower.startswith("c0=") or lower.startswith("c°="):
            try:
                standard_conc_M = _parse_standard_conc_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"C° set to {standard_conc_M:g} M")
            continue

        if lower.startswith("κ=") or lower.startswith("kappa="):
            try:
                kappa_global = _parse_kappa_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"κ set to {kappa_global:g}")
            continue

        # Initial conditions: [A]=1.0 or [A] = 1.0 or init: A=1.0
        if line.startswith("[") and "=" in line:
            try:
                initials_from_dsl.update(_parse_bracket_initials(line))
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            continue

        if lower.startswith("init:") or lower.startswith("initial:"):
            try:
                initials_from_dsl.update(_parse_init_directive(line))
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            continue

        # State and edge definitions - parse into state network
        if lower.startswith("state:"):
            _, rest = line.split(":", 1)
            name, state_kwargs = _parse_state_step(
                rest,
                energy_unit=energy_unit,
                line_number=line_no,
                line_content=raw,
            )
            try:
                state_network.add_state(name, **state_kwargs)
            except ValueError as exc:
                raise DSLError(str(exc), line_number=line_no, line_content=raw) from exc
            continue

        if lower.startswith("edge:"):
            _, rest = line.split(":", 1)
            a, b = _parse_edge_step(rest, line_number=line_no, line_content=raw)
            try:
                state_network.add_edge(a, b)
            except (KeyError, ValueError) as exc:
                raise DSLError(str(exc), line_number=line_no, line_content=raw) from exc
            continue

        if is_let_algebra_line(raw):
            algebra_lines.append(raw)
            continue

        if is_param_algebra_line(raw):
            algebra_lines.append(raw)
            continue

        # reaction: line
        if lower.startswith("reaction:"):
            _, rest = line.split(":", 1)
            steps.append(
                _parse_reaction_step(
                    rest,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    kappa_global=kappa_global,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        # equilibrium: line
        if lower.startswith("equilibrium:"):
            _, rest = line.split(":", 1)
            steps.append(
                _parse_equilibrium_step(
                    rest,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    kappa_global=kappa_global,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        # Fallback: Try to parse as bare arrow syntax (no "reaction:" or "equilibrium:" prefix)
        if _ARROW_RE.search(line):
            steps.append(
                _parse_bare_arrow_step(
                    line,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    kappa_global=kappa_global,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        if is_bare_assignment_algebra_line(raw):
            algebra_lines.append(raw)
            continue

        raise DSLError(f"unrecognized line: {line!r}", line_number=line_no, line_content=raw)

    if state_network.states() or state_network.edges():
        try:
            state_network.validate()
        except (TSDegreeError, ValueError) as exc:
            raise DSLError(str(exc)) from None

    if temperature_schedule is not None:
        notes.append(f"Temperature schedule detected: {temperature_schedule}")
    if intervention_schedule is not None:
        notes.append("Intervention schedule detected")

    return DSLIR(
        version=_DSL_IR_VERSION,
        energy_unit=str(energy_unit),
        temperature_K=float(temperature_K),
        standard_conc_M=float(standard_conc_M),
        kappa_global=float(kappa_global),
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
        state_network=state_network,
        steps=steps,
        algebra_lines=algebra_lines,
        initials_from_dsl=initials_from_dsl,
        notes=notes,
    )


def parse_dsl(text: str, *, units: UnitsModel | None = None) -> DSLResult:
    """
    Parse DSL text and compute preview lines.

    Parameters
    ----------
    text : str
        DSL content.
    units : UnitsModel | None
        Units context (energy unit, T, C0/p0). If None, defaults are used.

    Returns
    -------
    DSLResult
    """
    ir = _parse_dsl_ir(text, units=units)
    from .dsl_preview import build_step_previews

    previews = build_step_previews(
        ir.steps,
        temperature_K=ir.temperature_K,
        kappa_global=ir.kappa_global,
    )
    return DSLResult(
        previews=previews,
        notes=list(ir.notes),
        temperature_schedule=ir.temperature_schedule,
        intervention_schedule=ir.intervention_schedule,
        ir=ir,
    )


def parse_and_preview(text: str, *, units: UnitsModel | None = None) -> List[str]:
    """
    Convenience: return just the preview strings for GUI display.
    """
    res = parse_dsl(text, units=units)
    return [p.text for p in res.previews]


# ------------------------------ Mechanism building ---------------------------

@dataclass
class ParsedStep:
    """Internal representation of a parsed reaction step."""
    reactants: Dict[str, float]
    products: Dict[str, float]
    reversible: bool
    kf: float
    kr: Optional[float]
    model: str  # "Eyring" or "Arrhenius"
    is_equilibrium: bool  # True if from "equilibrium:" line
    arrhenius_A: Optional[float] = None
    arrhenius_Ea_J_per_mol: Optional[float] = None
    eyring_dG_act_J_per_mol: Optional[float] = None
    kappa: Optional[float] = None
    standard_conc_M: Optional[float] = None
    dG_eq_J_per_mol: Optional[float] = None
    Keq_input: Optional[float] = None
    explicit_rates: List[float] = field(default_factory=list)
    user_kf_explicit: bool = False
    user_kr_explicit: bool = False


def _parse_energy_unit_directive(line: str) -> str:
    from .common import normalize_energy_unit

    kv = _parse_keyvals(line)
    raw = kv.get("energy")
    if not raw:
        raise DSLError("energy= directive requires a value (kJ/mol, kcal/mol, or J/mol)")
    try:
        return normalize_energy_unit(raw)
    except ValueError:
        raise DSLError(
            f"energy must be 'kJ/mol', 'kcal/mol', or 'J/mol', got {raw!r}",
            examples=["energy=kJ/mol", "energy=kcal/mol", "energy=J/mol"],
        )


def _parse_temperature_directive(line: str) -> float:
    kv = _parse_keyvals(line)
    raw = kv.get("T")
    if not raw:
        raise DSLError("T= directive requires a numeric value")
    T = _float_or_none(raw)
    if T is None or T <= 0:
        raise DSLError("T must be positive")
    return float(T)


def _parse_standard_conc_directive(line: str) -> float:
    kv = _parse_keyvals(line)
    raw = kv.get("C0")
    if not raw:
        raise DSLError("C0= directive requires a numeric value")
    C0 = _float_or_none(raw)
    if C0 is None or C0 <= 0:
        raise DSLError("C0 must be positive")
    return float(C0)


def _parse_kappa_directive(line: str) -> float:
    kv = _parse_keyvals(line)
    raw = kv.get("κ")
    if not raw:
        raise DSLError("κ= directive requires a numeric value")
    kappa = _float_or_none(raw)
    if kappa is None or kappa <= 0:
        raise DSLError("κ must be positive")
    return float(kappa)


def _parse_bracket_initials(line: str) -> Dict[str, float]:
    initials: Dict[str, float] = {}
    for item in line.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("[") and "]" in item and "=" in item:
            bracket_end = item.index("]")
            species = item[1:bracket_end].strip()
            eq_pos = item.index("=", bracket_end)
            value_str = item[eq_pos + 1 :].strip()
            value = _float_or_none(value_str)
            if value is None:
                raise DSLError(f"Initial condition value must be numeric, got: {value_str}")
            initials[species] = float(value)
    return initials


def _parse_init_directive(line: str) -> Dict[str, float]:
    _, rest = line.split(":", 1)
    kv = _parse_keyvals(rest)
    initials: Dict[str, float] = {}
    for species, value_str in kv.items():
        value = _float_or_none(value_str)
        if value is None:
            raise DSLError(f"Initial condition value must be numeric for {species}, got: {value_str}")
        initials[species] = float(value)
    return initials


def _parse_state_step(
    rest: str,
    *,
    energy_unit: str,
    line_number: int,
    line_content: str,
) -> Tuple[str, Dict[str, object]]:
    from .kinetics import normalize_energy_to_J_per_mol

    rest = rest.strip()
    try:
        name, kv = _parse_state_rest(rest)
    except Exception as exc:
        raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc

    if not name:
        raise DSLError(
            "state: requires a name",
            examples=["state: GS1 ; kind=GS ; energy=0.0"],
            line_number=line_number, line_content=line_content,
        )

    kind = (kv.get("kind") or "GS").strip().upper()
    energy_unit_val = kv.get("energy_unit") or energy_unit
    energy_val_raw = kv.get("energy")
    ej = 0.0
    if energy_val_raw is not None and str(energy_val_raw).strip() != "":
        tok = str(energy_val_raw).strip().split()
        if len(tok) == 2 and tok[1] in ("kJ/mol", "kcal/mol", "J/mol"):
            v0 = _float_or_none(tok[0])
            if v0 is None:
                raise DSLError(
                    "state energy must be numeric",
                    examples=["state: TS1 ; energy=85.5 kJ/mol"],
                    line_number=line_number, line_content=line_content,
                )
            ej = normalize_energy_to_J_per_mol(float(v0), tok[1])
        else:
            v0 = _float_or_none(tok[0])
            if v0 is None:
                raise DSLError(
                    "state energy must be numeric",
                    examples=["state: TS1 ; energy=85.5"],
                    line_number=line_number, line_content=line_content,
                )
            ej = normalize_energy_to_J_per_mol(float(v0), cast(str, energy_unit_val))

    degeneracy_raw = _float_or_none(kv.get("degeneracy"))
    degeneracy = 1.0 if degeneracy_raw is None else float(degeneracy_raw)
    standard_state = kv.get("standard_state") or "C0"
    members = None
    if kv.get("members") is not None and str(kv.get("members")).strip() != "":
        try:
            members = _parse_members_expr(str(kv.get("members")))
        except Exception as exc:
            raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
    std_prod = None
    if kv.get("std") is not None and str(kv.get("std")).strip() != "":
        std_val = _extract_numeric_value(str(kv.get("std")))
        if std_val is None:
            raise DSLError("std must be numeric", line_number=line_number, line_content=line_content)
        std_prod = float(std_val)

    return str(name), {
        "kind": kind,
        "energy": (ej, "J/mol"),
        "degeneracy": degeneracy,
        "standard_state": standard_state,
        "members": members,
        "std_conc_product_M": std_prod,
    }


def _parse_edge_step(
    rest: str,
    *,
    line_number: Optional[int] = None,
    line_content: Optional[str] = None,
) -> Tuple[str, str]:
    rest = rest.strip()
    if "," in rest:
        parts = rest.split(",")
    elif "-" in rest:
        parts = rest.split("-")
    else:
        raise DSLError(f"edge must be 'A,B' or 'A-B', got: {rest}",
                       line_number=line_number, line_content=line_content)

    if len(parts) != 2:
        raise DSLError(f"edge must connect exactly 2 states, got: {rest}",
                       line_number=line_number, line_content=line_content)

    a = parts[0].strip()
    b = parts[1].strip()
    return a, b


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
        raise DSLError(
            "Per-reaction T= is not supported; use a global T= directive",
            line_number=line_number, line_content=line_content,
        )
    if "energy" in params:
        raise DSLError(
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
        raise DSLError(
            f"Unknown {step_label} parameter(s): {', '.join(sorted(unknown))}",
            line_number=line_number, line_content=line_content,
        )


def _resolve_Keq_from_params(
    params: Dict[str, str],
    *,
    energy_unit: str,
    T: float,
    line_number: Optional[int],
    line_content: Optional[str],
) -> Tuple[float, Optional[float]]:
    """Resolve equilibrium constant Keq from explicit K/Keq input or dG_eq=.

    Precondition: at least one of "Keq" or "dG_eq" must be present in params.
    Returns (Keq, dG_eqJ_or_None). Raises DSLError on numeric/overflow/underflow issues.
    """
    from .kinetics import K_from_deltaG_eq, normalize_energy_to_J_per_mol

    Keq = _float_or_none(params.get("Keq"))
    dG_eqJ: Optional[float] = None
    if Keq is None:
        dG_eq_val = _float_or_none(params["dG_eq"])
        if dG_eq_val is None:
            raise DSLError("dG_eq must be numeric", line_number=line_number, line_content=line_content)
        dG = normalize_energy_to_J_per_mol(dG_eq_val, energy_unit)
        dG_eqJ = dG
        try:
            Keq = K_from_deltaG_eq(dG, T)
        except OverflowError as exc:
            raise DSLError(
                "Equilibrium constant overflow (dG_eq too large for given T)",
                line_number=line_number, line_content=line_content,
            ) from exc
        if Keq <= 0:
            raise DSLError(
                "Equilibrium constant underflowed to zero (dG_eq too large for given T)",
                line_number=line_number, line_content=line_content,
            )
    return Keq, dG_eqJ


def _parse_reaction_like_step(
    *,
    stoich_part: str,
    params: Dict[str, str],
    energy_unit: str,
    T: float,
    C0: float,
    kappa_global: float,
    missing_eyring_message: str,
    line_number: int | None,
    line_content: str | None,
) -> ParsedStep:
    from .common import molecularity
    from .kinetics import arrhenius_rate, eyring_rate, normalize_energy_to_J_per_mol

    try:
        react, prod, arrow = _parse_stoich(stoich_part)
    except DSLError as exc:
        if exc.line_number is None:
            raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                           line_number=line_number, line_content=line_content) from exc
        raise
    reversible = arrow in ("<->", "<=>")

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _REACTION_KNOWN_KEYS, "reaction",
                           line_number=line_number, line_content=line_content)

    model = "Eyring"
    if "A" in params or "Ea" in params:
        model = "Arrhenius"
    kappa_raw = _float_or_none(params.get("κ"))
    if kappa_raw is not None:
        kappa = kappa_raw
    else:
        kappa = kappa_global
    n = molecularity(react)

    kf = _float_or_none(params.get("kf") or params.get("k"))
    kr = _float_or_none(params.get("kr"))
    if params.get("kf") or params.get("k"):
        _validate_rate_or_K(kf, "kf" if "kf" in params else "k",
                            line_number=line_number, line_content=line_content)
    if params.get("kr"):
        _validate_rate_or_K(kr, "kr", line_number=line_number, line_content=line_content)
    if "Keq" in params:
        _Keq_early = _float_or_none(params["Keq"])
        if _Keq_early is not None and _Keq_early <= 0:
            raise DSLError(
                f"Keq must be positive, got {_Keq_early}",
                line_number=line_number, line_content=line_content,
            )
    arrhenius_A = None
    arrhenius_EaJ = None
    eyring_dGJ = None
    dG_eqJ_for_step = None
    Keq_input = None
    explicit_rates: List[float] = []

    if model == "Arrhenius":
        A = _float_or_none(params.get("A"))
        Ea = params.get("Ea")
        if A is None or Ea is None:
            raise DSLError(
                "Arrhenius requires A and Ea",
                examples=["A -> B ; A=1e13 ; Ea=50"],
                line_number=line_number, line_content=line_content,
            )
        if A <= 0:
            raise DSLError(
                f"Arrhenius pre-exponential factor A must be positive, got {A}",
                line_number=line_number, line_content=line_content,
            )
        Ea_val = _float_or_none(Ea)
        if Ea_val is None:
            raise DSLError("Ea must be numeric", line_number=line_number, line_content=line_content)
        EaJ = normalize_energy_to_J_per_mol(Ea_val, energy_unit)
        arrhenius_A = A
        arrhenius_EaJ = EaJ
        try:
            kf = arrhenius_rate(A, EaJ, T)
        except OverflowError as exc:
            raise DSLError(
                "Arrhenius rate computation overflowed (A and Ea produce an unrepresentable rate)",
                line_number=line_number, line_content=line_content,
            ) from exc
        explicit_rates.append(kf)
        if reversible and kr is None:
            if "Keq" in params or "dG_eq" in params:
                Keq, dG_eqJ_for_step = _resolve_Keq_from_params(
                    params, energy_unit=energy_unit, T=T,
                    line_number=line_number, line_content=line_content,
                )
                kr = kf / Keq
                Keq_input = Keq if "Keq" in params else None
            else:
                raise DSLError(
                    "reversible Arrhenius step needs kr or Keq/dG_eq",
                    examples=["A <-> B ; A=1e10 ; Ea=50 ; K=2.0"],
                    line_number=line_number, line_content=line_content,
                )
    else:
        # Eyring
        if kf is None:
            if "dG_act" not in params:
                raise DSLError(
                    missing_eyring_message,
                    examples=["A -> B ; dG_act=75.5", "A -> B ; k=1.5"],
                    line_number=line_number, line_content=line_content,
                )
            dG_act_val = _float_or_none(params["dG_act"])
            if dG_act_val is None:
                raise DSLError("dG_act must be numeric", line_number=line_number, line_content=line_content)
            dGJ = normalize_energy_to_J_per_mol(dG_act_val, energy_unit)
            eyring_dGJ = dGJ
            if kappa <= 0:
                raise DSLError(
                    f"Per-step kappa must be positive for Eyring rate computation, got {kappa}",
                    line_number=line_number, line_content=line_content,
                )
            try:
                kf = eyring_rate(dGJ, T, kappa=kappa, molecularity=n, standard_conc_M=C0)
            except OverflowError as exc:
                raise DSLError(
                    "Eyring rate computation overflowed (dG_act too large for given T)",
                    line_number=line_number, line_content=line_content,
                ) from exc
            explicit_rates.append(kf)
        if reversible and kr is None:
            if "Keq" in params or "dG_eq" in params:
                Keq, dG_eqJ_for_step = _resolve_Keq_from_params(
                    params, energy_unit=energy_unit, T=T,
                    line_number=line_number, line_content=line_content,
                )
                kr = kf / Keq
                Keq_input = Keq if "Keq" in params else None
            else:
                dG_eq_fallback_J_per_mol = None
                dG_eq_raw = params.get("dG_eq")
                if dG_eq_raw is not None:
                    dG_eq_val = _float_or_none(dG_eq_raw)
                    if dG_eq_val is None:
                        raise DSLError("dG_eq must be numeric", line_number=line_number, line_content=line_content)
                    dG_eq_fallback_J_per_mol = normalize_energy_to_J_per_mol(dG_eq_val, energy_unit)
                fe = _derive_equilibrium_rates_with_context(
                    Keq=_float_or_none(params.get("Keq")),
                    dG_eq_J_per_mol=dG_eq_fallback_J_per_mol,
                    T=T,
                    explicit_rates=[kf] if kf is not None else None,
                    line_number=line_number,
                    line_content=line_content,
                )
                kr = fe.kr

    if kr is not None:
        explicit_rates.append(kr)

    return ParsedStep(
        reactants=react,
        products=prod,
        reversible=reversible,
        kf=kf,
        kr=kr,
        model=model,
        is_equilibrium=False,
        arrhenius_A=arrhenius_A,
        arrhenius_Ea_J_per_mol=arrhenius_EaJ,
        eyring_dG_act_J_per_mol=eyring_dGJ,
        kappa=kappa,
        standard_conc_M=C0,
        dG_eq_J_per_mol=dG_eqJ_for_step,
        Keq_input=Keq_input,
        explicit_rates=explicit_rates,
        user_kf_explicit=bool(
            params.get("kf") or params.get("k") or params.get("A") or params.get("Ea") or params.get("dG_act")
        ),
        user_kr_explicit=bool(params.get("kr")),
    )


def _parse_reaction_step(
    rest: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    kappa_global: float,
    line_number: int,
    line_content: str,
) -> ParsedStep:
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
        kappa_global=kappa_global,
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
    kappa_global: float,
    line_number: int,
    line_content: str,
) -> ParsedStep:
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
        kappa_global=kappa_global,
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
    kappa_global: float,
    line_number: int,
    line_content: str,
) -> ParsedStep:
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
    except DSLError as exc:
        if exc.line_number is None:
            raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                           line_number=line_number, line_content=line_content) from exc
        raise
    if arrow not in ("<->", "<=>"):
        raise DSLError(
            "equilibrium must use '<->' or '<=>'",
            examples=["equilibrium: A <-> B ; Keq=2.0 ; kf=10"],
            line_number=line_number, line_content=line_content,
        )
    _ = molecularity(react)

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _EQUILIBRIUM_KNOWN_KEYS, "equilibrium",
                           line_number=line_number, line_content=line_content)

    has_explicit_Keq = "Keq" in params
    Keq = _float_or_none(params.get("Keq"))
    if has_explicit_Keq and Keq is None:
        raise DSLError(
            "Keq must be numeric",
            line_number=line_number,
            line_content=line_content,
        )
    if has_explicit_Keq and Keq is not None:
        _validate_rate_or_K(Keq, "Keq", line_number=line_number, line_content=line_content)
    dG_eqJ = None

    exp_rates: List[float] = []
    kf_explicit = None
    kr_explicit = None
    if "kf" in params:
        kfv = _float_or_none(params["kf"])
        if kfv is None:
            raise DSLError("kf must be numeric", line_number=line_number, line_content=line_content)
        _validate_rate_or_K(kfv, "kf", line_number=line_number, line_content=line_content)
        kf_explicit = kfv
        exp_rates.append(kfv)
    if "kr" in params:
        krv = _float_or_none(params["kr"])
        if krv is None:
            raise DSLError("kr must be numeric", line_number=line_number, line_content=line_content)
        _validate_rate_or_K(krv, "kr", line_number=line_number, line_content=line_content)
        kr_explicit = krv
        exp_rates.append(krv)

    if has_explicit_Keq and kf_explicit is None and kr_explicit is None:
        raise DSLError(
            "equilibrium with Keq=... requires at least one of kf or kr to anchor the rates",
            suggestion="Provide either kf=...; Keq=... or kr=...; Keq=... (or specify both kf and kr without Keq).",
            examples=[
                "equilibrium: A <-> B; kf=10.0; Keq=5.0",
                "equilibrium: A <-> B; kr=2.0; Keq=5.0",
                "equilibrium: A <-> B; kf=4.0; kr=2.0",
            ],
            line_number=line_number,
            line_content=line_content,
        )

    if has_explicit_Keq and kf_explicit is not None and kr_explicit is not None:
        if abs(kr_explicit) < 1e-30:
            raise DSLError(
                "kr must be non-zero when validating Keq against kf/kr",
                line_number=line_number,
                line_content=line_content,
            )
        implied_K = float(kf_explicit) / float(kr_explicit)
        tol_rel = 1e-6
        tol_abs = 1e-12
        diff = abs(implied_K - float(Keq))
        scale = max(abs(implied_K), abs(float(Keq)), 1.0)
        if diff > (tol_abs + tol_rel * scale):
            raise DSLError(
                f"Inconsistent equilibrium parameters: Keq={float(Keq):.6g} but kf/kr={implied_K:.6g}",
                suggestion="Adjust Keq or kf/kr so that Keq ≈ kf/kr (within tolerance).",
                line_number=line_number,
                line_content=line_content,
            )

    if Keq is None and kf_explicit is not None and kr_explicit is not None and kr_explicit != 0:
        Keq = kf_explicit / kr_explicit

    if Keq is None:
        dG_eq = params.get("dG_eq") or params.get("dg_eq")
        if dG_eq is None:
            raise DSLError(
                "equilibrium requires Keq, dG_eq, or both kf and kr. "
                "Examples: 'Keq=2.0' or 'kf=1.5; kr=0.25' or 'dG_eq=-10 kJ/mol'",
                line_number=line_number,
                line_content=line_content,
            )
        dG_eq_val = _float_or_none(dG_eq)
        if dG_eq_val is None:
            raise DSLError("dG_eq must be numeric", line_number=line_number, line_content=line_content)
        dG_eqJ = normalize_energy_to_J_per_mol(dG_eq_val, energy_unit)

    fe = _derive_equilibrium_rates_with_context(
        Keq=Keq,
        dG_eq_J_per_mol=dG_eqJ,
        T=T,
        explicit_rates=exp_rates or None,
        line_number=line_number,
        line_content=line_content,
    )
    kf = fe.kf if kf_explicit is None else kf_explicit
    kr = fe.kr if kr_explicit is None else kr_explicit

    return ParsedStep(
        reactants=react,
        products=prod,
        reversible=True,
        kf=kf,
        kr=kr,
        model="Eyring",
        is_equilibrium=True,
        eyring_dG_act_J_per_mol=None,
        kappa=kappa_global,
        standard_conc_M=C0,
        dG_eq_J_per_mol=dG_eqJ,
        Keq_input=(Keq if has_explicit_Keq else None),
        explicit_rates=exp_rates,
        user_kf_explicit=("kf" in params),
        user_kr_explicit=("kr" in params),
    )


def parse_dsl_to_mechanism(
    text: str,
    *,
    initials: Optional[Dict[str, float]] = None,
    units: UnitsModel | None = None
) -> Mechanism:
    """
    Parse DSL text and build a Mechanism object ready for simulation.

    This function uses the full DSL parser with all advanced features
    (Eyring/Arrhenius, state networks, fast equilibrium, etc.) and
    converts the result into a Mechanism object.

    Parameters
    ----------
    text : str
        DSL content supporting all features:
        - reaction: A -> B; dG_act=80, energy=kJ/mol
        - reaction: A + B -> C; Ea=50, A=1e13, energy=kJ/mol
        - equilibrium: A <-> B; K=4.0
        - equilibrium: A <-> B; dG_eq=-5, energy=kJ/mol
        - Global settings: T=310, energy=kJ/mol, κ=0.8, C0=1.0
        - State networks: state: A, kind=GS, energy=0
        - Edges: edge: A,TS1
    initials : dict, optional
        Initial concentrations {species: value}. Treated as input-only (this
        function does not mutate the caller's mapping). When the DSL also
        specifies an initial concentration for the same species, the DSL value
        takes precedence.
    units : UnitsModel, optional
        Units context (energy unit, T, C0). If None, defaults are used.

    Returns
    -------
    Mechanism
        Mechanism object populated with species, reactions, and equilibria

    Raises
    ------
    DSLError
        If parsing or validation fails

    Examples
    --------
    >>> dsl_text = '''
    ... T=310
    ... reaction: A -> B; dG_act=80, energy=kJ/mol
    ... equilibrium: B <-> C; K=4.0
    ... '''
    >>> mech = parse_dsl_to_mechanism(dsl_text, initials={'A': 1.0, 'B': 0.0, 'C': 0.0})
    >>> print(mech.species_names())
    ['A', 'B', 'C']
    """
    ir = _parse_dsl_ir(text, units=units)
    from .dsl_build import build_mechanism_from_ir

    return build_mechanism_from_ir(ir, initials=initials)
