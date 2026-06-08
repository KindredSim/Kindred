"""
Computational Mode: convert absolute computed free energies into Kindred's ΔG-based energy-mode DSL.

This module is intentionally Qt-free so it can be unit-tested without launching a GUI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import scipy.constants

from kindred.core.constants import R
from kindred.core.simulator.common import normalize_energy_unit

__all__ = [
    "COMP_BLOCK_START",
    "COMP_BLOCK_END",
    "GENERATED_BLOCK_START",
    "GENERATED_BLOCK_END",
    "CompSpec",
    "CompiledComp",
    "hartree_to_jmol",
    "extract_marked_block",
    "upsert_marked_block",
    "upsert_computational_mode_blocks",
    "parse_comp_block",
    "compile_comp_spec",
]


COMP_BLOCK_START = "# === Computational Mode ==="
COMP_BLOCK_END = "# === End Computational Mode ==="
GENERATED_BLOCK_START = "# === Generated from Computational Mode ==="
GENERATED_BLOCK_END = "# === End Generated from Computational Mode ==="


_SPECIES_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ARROW_RE = re.compile(r"<=>|<->|->")
_SPECIES_TERM_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*\*?\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*$")


def hartree_to_jmol(x_hartree: float) -> float:
    """
    Convert Hartree (Eh) to J/mol.

    Uses SciPy's CODATA constants:
    - Hartree energy [J] per particle
    - Avogadro constant [1/mol]
    """
    return float(x_hartree) * float(scipy.constants.value("Hartree energy")) * float(scipy.constants.N_A)


def _energy_to_jmol(value: float, unit: str) -> float:
    unit_n = normalize_energy_unit(unit, allow_hartree=True)
    if unit_n == "hartree":
        return hartree_to_jmol(float(value))
    if unit_n == "J/mol":
        return float(value)
    if unit_n == "kJ/mol":
        return float(value) * 1000.0
    if unit_n == "kcal/mol":
        return float(value) * 4184.0
    raise ValueError(f"unsupported energy unit {unit!r}")


def _parse_float_token(s: str) -> float:
    try:
        x = float(str(s).strip())
    except Exception as e:
        raise ValueError(f"expected a number, got {s!r}") from e
    if not math.isfinite(x):
        raise ValueError("value must be finite")
    return float(x)


def _parse_float_with_optional_unit(expr: str, *, default_unit: str) -> Tuple[float, str]:
    expr = str(expr or "").strip()
    if not expr:
        raise ValueError("missing value")
    parts = expr.split()
    if len(parts) == 1:
        return _parse_float_token(parts[0]), str(default_unit)
    if len(parts) == 2:
        return _parse_float_token(parts[0]), str(parts[1])
    raise ValueError(f"invalid value {expr!r}")


def _parse_conc_M(expr: str) -> float:
    v, unit = _parse_float_with_optional_unit(expr, default_unit="M")
    unit_s = str(unit).strip()
    if unit_s not in {"M", "m"}:
        raise ValueError(f"expected concentration unit 'M', got {unit!r}")
    if v <= 0.0:
        raise ValueError("concentration must be positive")
    return float(v)


def _parse_pressure_Pa(expr: str) -> float:
    v, unit = _parse_float_with_optional_unit(expr, default_unit="atm")
    u = str(unit).strip().lower()
    if u == "pa":
        P = float(v)
    elif u == "atm":
        P = float(v) * float(scipy.constants.atm)
    else:
        raise ValueError(f"expected pressure unit 'atm' or 'Pa', got {unit!r}")
    if not math.isfinite(P) or P <= 0.0:
        raise ValueError("pressure must be positive and finite")
    return float(P)


def _parse_temperature_K(expr: str) -> float:
    v, unit = _parse_float_with_optional_unit(expr, default_unit="K")
    u = str(unit).strip().lower()
    if u != "k":
        raise ValueError(f"expected temperature unit 'K', got {unit!r}")
    if v <= 0.0:
        raise ValueError("temperature must be positive")
    return float(v)


def _validate_species_name(name: str) -> str:
    n = str(name or "").strip()
    if not n:
        raise ValueError("species name cannot be empty")
    if not _SPECIES_NAME_RE.match(n):
        raise ValueError(f"invalid species name {n!r}")
    return n


def _split_keyvals_preserving_units(tokens: List[str]) -> Dict[str, str]:
    """
    Parse key=value pairs from a token list.

    Supports 2-token values for units, e.g. ["std=19.2", "M"] -> {"std": "19.2 M"}.
    """
    out: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = str(tokens[i]).strip()
        if not tok:
            i += 1
            continue
        if "=" not in tok:
            raise ValueError(f"expected key=value, got {tok!r}")
        key, val = tok.split("=", 1)
        key = key.strip()
        val = val.strip()
        if i + 1 < len(tokens):
            nxt = str(tokens[i + 1]).strip()
            if nxt and ("=" not in nxt) and nxt in {"M", "m", "K", "k", "atm", "Pa", "pa", "hartree", "Eh"}:
                val = f"{val} {nxt}"
                i += 1
        out[str(key)] = str(val)
        i += 1
    return out


def _parse_stoich_side(side: str) -> Dict[str, float]:
    side = str(side or "").strip()
    if not side:
        raise ValueError("empty stoichiometry side")
    parts = [p.strip() for p in side.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty stoichiometry side")
    out: Dict[str, float] = {}
    for raw in parts:
        term = raw.strip()
        m = _SPECIES_TERM_RE.match(term)
        if not m:
            raise ValueError(f"invalid term {raw!r}")
        coeff_s, name_s = m.groups()
        coeff = float(coeff_s) if coeff_s is not None else 1.0
        if not math.isfinite(coeff):
            raise ValueError("stoichiometric coefficients must be finite")
        if coeff < 0.0:
            raise ValueError("stoichiometric coefficients must be non-negative")
        if coeff == 0.0:
            continue
        name = _validate_species_name(name_s)
        out[name] = out.get(name, 0.0) + float(coeff)
    if not out:
        raise ValueError("empty stoichiometry side")
    return out


def _parse_equation(eq: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    eq = str(eq or "").strip()
    m = _ARROW_RE.search(eq)
    if not m:
        raise ValueError("missing reaction arrow (->, <->, or <=>)")
    arrow = m.group(0)
    if arrow not in ("<->", "<=>"):
        raise ValueError("computational mode reactions must be reversible: use '<->' or '<=>'")
    lhs = eq[: m.start()].strip()
    rhs = eq[m.end() :].strip()
    return _parse_stoich_side(lhs), _parse_stoich_side(rhs)


def _fmt_coeff(coeff: float) -> str:
    return f"{float(coeff):.12g}"


def _canonical_side(stoich: Dict[str, float]) -> str:
    parts: List[str] = []
    for name in sorted(stoich.keys()):
        coeff = float(stoich[name])
        if abs(coeff - 1.0) < 1e-12:
            parts.append(f"{name}")
        else:
            parts.append(f"{_fmt_coeff(coeff)}{name}")
    return "+".join(parts)


def _sum_energy(stoich: Dict[str, float], energies: Dict[str, float]) -> float:
    total = 0.0
    for name, coeff in stoich.items():
        total += float(coeff) * float(energies[name])
    return float(total)


def _std_product(stoich: Dict[str, float], std_M: Dict[str, float]) -> float:
    prod = 1.0
    for name, coeff in stoich.items():
        prod *= float(std_M[name]) ** float(coeff)
    return float(prod)


@dataclass(frozen=True)
class CompSpecies:
    name: str
    kind: str  # "GS" | "TS"
    G_value: float
    degeneracy: float = 1.0
    std_M: Optional[float] = None
    cref_M: Optional[float] = None


@dataclass(frozen=True)
class CompReaction:
    reactants: Dict[str, float]
    products: Dict[str, float]
    via_ts: Optional[str] = None
    fast_k: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reactants", dict(self.reactants))
        object.__setattr__(self, "products", dict(self.products))


@dataclass(frozen=True)
class CompSpec:
    temperature_K: float = 298.15
    pressure_Pa: float = float(scipy.constants.atm)
    energy_unit: str = "hartree"
    std_default_M: float = 1.0
    kfast_default: float = 1e9
    kfast_by_order: Dict[int, float] = field(default_factory=dict)
    species: Dict[str, CompSpecies] = field(default_factory=dict)
    reactions: List[CompReaction] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kfast_by_order", dict(self.kfast_by_order))
        object.__setattr__(self, "species", dict(self.species))
        object.__setattr__(self, "reactions", list(self.reactions))


@dataclass(frozen=True)
class CompiledComp:
    spec: CompSpec
    species_G_std_J_per_mol: Dict[str, float]
    generated_reaction_dsl: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_G_std_J_per_mol", dict(self.species_G_std_J_per_mol))


def extract_marked_block(text: str, *, start_marker: str, end_marker: str) -> Optional[str]:
    """
    Return the block body between markers (excluding marker lines), or None if not present.

    If the start marker exists but the end marker is missing, the block runs to EOF.
    """
    lines = str(text or "").splitlines()
    start_idx = None
    for i, raw in enumerate(lines):
        if raw.strip() == str(start_marker).strip():
            start_idx = i
            break
    if start_idx is None:
        return None
    end_idx = None
    for j in range(start_idx + 1, len(lines)):
        if lines[j].strip() == str(end_marker).strip():
            end_idx = j
            break
    body_lines = lines[start_idx + 1 : (end_idx if end_idx is not None else len(lines))]
    return "\n".join(body_lines).rstrip("\n")


def upsert_marked_block(
    text: str,
    *,
    start_marker: str,
    end_marker: str,
    body: str,
    blank_line_before: bool = True,
) -> str:
    """
    Replace a delimited block if present, else append a new block.
    """
    start_marker = str(start_marker).rstrip("\n")
    end_marker = str(end_marker).rstrip("\n")
    body_lines = [ln.rstrip("\n") for ln in str(body or "").strip("\n").splitlines()]
    block_lines = [start_marker] + body_lines + [end_marker]

    lines = str(text or "").splitlines()
    start_positions = [i for i, raw in enumerate(lines) if raw.strip() == start_marker.strip()]
    if start_positions:
        # Replace the *first* well-formed block and delete any additional occurrences.
        first_start = int(start_positions[0])
        end_idx = None
        for j in range(first_start + 1, len(lines)):
            if lines[j].strip() == end_marker.strip():
                end_idx = j
                break
        if end_idx is None:
            end_idx = len(lines) - 1
        new_lines = lines[:first_start] + block_lines + lines[end_idx + 1 :]
        # Remove additional blocks if present (defensive).
        out: List[str] = []
        i = 0
        while i < len(new_lines):
            if new_lines[i].strip() == start_marker.strip() and i != first_start:
                # Skip until matching end marker.
                i += 1
                while i < len(new_lines) and new_lines[i].strip() != end_marker.strip():
                    i += 1
                if i < len(new_lines) and new_lines[i].strip() == end_marker.strip():
                    i += 1
                continue
            out.append(new_lines[i])
            i += 1
        return "\n".join(out).rstrip("\n") + "\n"

    # Append block.
    prefix = "\n".join(lines).rstrip("\n")
    if not prefix:
        return "\n".join(block_lines).rstrip("\n") + "\n"
    joiner = "\n\n" if blank_line_before else "\n"
    return (prefix + joiner + "\n".join(block_lines)).rstrip("\n") + "\n"


def upsert_computational_mode_blocks(
    reaction_dsl_text: str,
    *,
    comp_body: str,
    generated_body: str,
) -> str:
    """
    Upsert both Computational Mode blocks in the Reaction DSL text.
    """
    updated = upsert_marked_block(
        reaction_dsl_text,
        start_marker=COMP_BLOCK_START,
        end_marker=COMP_BLOCK_END,
        body=comp_body,
        blank_line_before=True,
    )
    updated = upsert_marked_block(
        updated,
        start_marker=GENERATED_BLOCK_START,
        end_marker=GENERATED_BLOCK_END,
        body=generated_body,
        blank_line_before=True,
    )
    return updated


def parse_comp_block(comp_block_body: str) -> CompSpec:
    """
    Parse Computational Mode `comp:` lines (body only, excluding markers).

    Unknown directives raise ValueError for deterministic behavior.
    """
    temperature_K = 298.15
    pressure_Pa = float(scipy.constants.atm)
    energy_unit = "hartree"
    std_default_M = 1.0
    kfast_default = 1e9
    kfast_by_order: Dict[int, float] = {}
    species: Dict[str, CompSpecies] = {}
    reactions: List[CompReaction] = []

    for raw in str(comp_block_body or "").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.lower().startswith("comp:"):
            raise ValueError(f"expected comp: line, got {raw!r}")
        line = stripped[len("comp:") :].strip()
        if not line:
            continue

        # Global directives: <key> = <value>
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            if key in {"t", "temp", "temperature"}:
                temperature_K = _parse_temperature_K(val)
                continue
            if key == "pressure":
                pressure_Pa = _parse_pressure_Pa(val)
                continue
            if key == "energy_unit":
                energy_unit = normalize_energy_unit(val, allow_hartree=True)
                continue
            if key == "std_default":
                std_default_M = _parse_conc_M(val)
                continue
            if key == "kfast_default":
                kfast_default = _parse_float_token(val)
                if kfast_default <= 0.0:
                    raise ValueError("kfast_default must be positive")
                continue
            if key.startswith("kfast_"):
                order_s = key[len("kfast_") :].strip()
                if not order_s.isdigit():
                    raise ValueError(f"invalid kfast order key {key!r}")
                order = int(order_s)
                v = _parse_float_token(val)
                if v <= 0.0:
                    raise ValueError("kfast_m must be positive")
                kfast_by_order[int(order)] = float(v)
                continue
            raise ValueError(f"unknown computational mode directive {key!r}")

        # Entity lines: species / rxn / channel
        parts = line.split()
        head = parts[0].strip().lower()
        if head == "species":
            if len(parts) < 2:
                raise ValueError("comp: species requires a name")
            name = _validate_species_name(parts[1])
            kv = _split_keyvals_preserving_units(parts[2:])
            kind = str(kv.get("type") or kv.get("kind") or "GS").strip().upper()
            if kind not in {"GS", "TS"}:
                raise ValueError("species type must be GS or TS")
            if "G" not in kv and "g" not in kv:
                raise ValueError("species requires G=<value>")
            G_raw = kv.get("G") if "G" in kv else kv.get("g")
            if G_raw is None:
                raise ValueError("species requires G=<value>")
            G_val = _parse_float_token(str(G_raw).split()[0])
            degeneracy = _parse_float_token(kv.get("degeneracy") or "1")
            if degeneracy <= 0.0:
                raise ValueError("degeneracy must be positive")
            std_M = _parse_conc_M(kv["std"]) if "std" in kv else None
            cref_M = _parse_conc_M(kv["cref"]) if "cref" in kv else None
            species[name] = CompSpecies(
                name=name,
                kind=kind,
                G_value=float(G_val),
                degeneracy=float(degeneracy),
                std_M=std_M,
                cref_M=cref_M,
            )
            continue

        if head in {"rxn", "reaction", "channel"}:
            rest = line[len(parts[0]) :].strip()
            via_ts = None
            fast_k = None

            # Extract optional "via TS" for channels.
            m_via = re.search(r"\s+via\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", rest, flags=re.IGNORECASE)
            eq_part = rest
            if m_via is not None:
                via_ts = _validate_species_name(m_via.group(1))
                eq_part = rest[: m_via.start()].strip()

            # Remaining keyvals (e.g., fast_k=...) are supported only on rxn lines.
            tokens = eq_part.split()
            kv_tokens: List[str] = []
            eq_tokens: List[str] = []
            for t in tokens:
                if ("=" in t) and ("<=>" not in t):
                    kv_tokens.append(t)
                else:
                    eq_tokens.append(t)
            eq_str = " ".join(eq_tokens).strip()
            reactants, products = _parse_equation(eq_str)
            if kv_tokens:
                kv = _split_keyvals_preserving_units(kv_tokens)
                if "fast_k" in kv:
                    fast_k = _parse_float_token(kv["fast_k"])
                    if fast_k <= 0.0:
                        raise ValueError("fast_k must be positive")

            if head != "channel" and via_ts is not None:
                # Accept "rxn ... via TS" as a synonym for channel.
                pass

            reactions.append(
                CompReaction(
                    reactants=reactants,
                    products=products,
                    via_ts=via_ts,
                    fast_k=fast_k,
                )
            )
            continue

        raise ValueError(f"unknown comp entry {parts[0]!r}")

    if not (math.isfinite(temperature_K) and temperature_K > 0.0):
        raise ValueError("T must be positive and finite")
    if not (math.isfinite(pressure_Pa) and pressure_Pa > 0.0):
        raise ValueError("pressure must be positive and finite")
    if not (math.isfinite(std_default_M) and std_default_M > 0.0):
        raise ValueError("std_default must be positive and finite")

    return CompSpec(
        temperature_K=float(temperature_K),
        pressure_Pa=float(pressure_Pa),
        energy_unit=str(energy_unit),
        std_default_M=float(std_default_M),
        kfast_default=float(kfast_default),
        kfast_by_order=dict(kfast_by_order),
        species=dict(species),
        reactions=list(reactions),
    )


def compile_comp_spec(spec: CompSpec, *, output_energy_unit: str = "kJ/mol") -> CompiledComp:
    """
    Compile a parsed CompSpec to generated Reaction DSL text.

    - Computes standard-state corrected species free energies in J/mol.
    - Generates a replaceable block containing:
        - global energy+T directives
        - inline state network (for TS channels) with members/std metadata
        - explicit equilibrium lines for GS–GS fast equilibria (no TS)
    """
    # Basic validation
    sp = dict(spec.species or {})
    if not sp:
        raise ValueError("no species defined")
    T = float(spec.temperature_K)
    P = float(spec.pressure_Pa)
    energy_unit = str(spec.energy_unit)
    std_default = float(spec.std_default_M)

    out_energy_unit = normalize_energy_unit(output_energy_unit, allow_hartree=True)
    if out_energy_unit not in {"kJ/mol", "kcal/mol"}:
        raise ValueError("output_energy_unit must be 'kJ/mol' or 'kcal/mol'")
    from kindred.core.units import UnitsModel

    out_units = UnitsModel(energy_unit=out_energy_unit)

    # Reference concentration: ideal gas at given T, P (mol/L)
    c_gas_M = P / (R * T) / 1000.0
    if not (math.isfinite(c_gas_M) and c_gas_M > 0.0):
        raise ValueError("invalid gas reference concentration")

    species_std_M: Dict[str, float] = {}
    species_deg: Dict[str, float] = {}
    for name, entry in sp.items():
        species_std_M[str(name)] = float(entry.std_M) if entry.std_M is not None else std_default
        species_deg[str(name)] = float(entry.degeneracy) if entry.degeneracy is not None else 1.0

    # Standard-state correction for each species.
    species_G_std_J: Dict[str, float] = {}
    for name, entry in sp.items():
        std_M = species_std_M[name]
        cref_M = float(entry.cref_M) if entry.cref_M is not None else float(c_gas_M)
        if not (math.isfinite(std_M) and std_M > 0.0):
            raise ValueError("std must be positive and finite")
        if not (math.isfinite(cref_M) and cref_M > 0.0):
            raise ValueError("cref must be positive and finite")

        G_input_J = _energy_to_jmol(float(entry.G_value), energy_unit)
        correction = R * T * math.log(float(std_M) / float(cref_M))
        species_G_std_J[name] = float(G_input_J + correction)

    # Compile channels and fast equilibria.
    reactions = list(spec.reactions or [])
    if not reactions:
        raise ValueError("no reactions/channels defined")

    # Collect unique GS sides used in TS channels (reactant/product complexes).
    gs_sides: Dict[str, Dict[str, int]] = {}
    ts_states: Dict[str, str] = {}  # generated TS state name -> underlying TS species name
    edges: List[Tuple[str, str]] = []
    state_energy_J: Dict[str, float] = {}
    state_kind: Dict[str, str] = {}
    state_members: Dict[str, Dict[str, int]] = {}
    state_std_prod: Dict[str, float] = {}
    state_deg: Dict[str, float] = {}

    equilibrium_lines: List[str] = []

    def _ensure_gs_state(stoich: Dict[str, int]) -> str:
        key = _canonical_side(stoich)
        if key not in gs_sides:
            gs_sides[key] = dict(stoich)
        return key

    def _stoich_to_positive_ints(stoich: Dict[str, float], *, context: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, coeff in (stoich or {}).items():
            c = float(coeff)
            if not math.isfinite(c) or c <= 0.0:
                raise ValueError(f"{context} stoichiometry must use positive finite coefficients")
            n = int(round(c))
            if abs(c - float(n)) > 1e-12:
                raise ValueError(f"{context} stoichiometry must use positive integers")
            out[str(name)] = int(n)
        return out

    # Helper: compute fast k default per order.
    def _kfast_for_order(order: int) -> float:
        if int(order) in (spec.kfast_by_order or {}):
            return float(spec.kfast_by_order[int(order)])
        return float(spec.kfast_default)

    for rxn in reactions:
        # Validate referenced species.
        for n in list(rxn.reactants.keys()) + list(rxn.products.keys()):
            if n not in sp:
                raise ValueError(f"unknown species {n!r} in reaction stoichiometry")

        if rxn.via_ts:
            ts_name = str(rxn.via_ts)
            if ts_name not in sp:
                raise ValueError(f"unknown TS {ts_name!r}")
            if str(sp[ts_name].kind).upper() != "TS":
                raise ValueError(f"channel via {ts_name!r} requires species type=TS")

            react_int = _stoich_to_positive_ints(rxn.reactants, context="channel")
            prod_int = _stoich_to_positive_ints(rxn.products, context="channel")
            gs_react = _ensure_gs_state(react_int)
            gs_prod = _ensure_gs_state(prod_int)

            # Generate a unique TS state name if the TS is reused across channels.
            ts_state_name = ts_name
            if ts_state_name in ts_states and ts_states[ts_state_name] != ts_name:
                raise ValueError("internal TS state collision")
            if ts_state_name in ts_states:
                # TS already used: create deterministic disambiguation.
                ts_state_name = f"{ts_name}__{gs_react}__{gs_prod}"
            ts_states[ts_state_name] = ts_name

            edges.append((gs_react, ts_state_name))
            edges.append((ts_state_name, gs_prod))
        else:
            # Fast equilibrium (no TS): explicit equilibrium line in generated DSL.
            order = float(sum(float(v) for v in rxn.reactants.values()))
            if not math.isfinite(order) or order <= 0.0:
                raise ValueError("invalid molecularity for fast equilibrium")
            m_int = int(round(order))
            use_m = abs(order - float(m_int)) <= 1e-9 and m_int >= 1
            kf = float(rxn.fast_k) if rxn.fast_k is not None else float(_kfast_for_order(m_int) if use_m else spec.kfast_default)
            if not (math.isfinite(kf) and kf > 0.0):
                raise ValueError("fast equilibrium kf must be positive and finite")

            G_react = _sum_energy(rxn.reactants, species_G_std_J)
            G_prod = _sum_energy(rxn.products, species_G_std_J)
            dG_eq = float(G_prod - G_react)
            K = float(math.exp(-dG_eq / (R * T)))
            if not (math.isfinite(K) and K > 0.0):
                raise ValueError("computed K must be positive and finite")

            std_react = _std_product(rxn.reactants, species_std_M)
            std_prod = _std_product(rxn.products, species_std_M)
            if not (math.isfinite(std_react) and std_react > 0.0 and math.isfinite(std_prod) and std_prod > 0.0):
                raise ValueError("invalid std products for fast equilibrium")

            Kc = float(K * (std_prod / std_react))
            if not (math.isfinite(Kc) and Kc > 0.0):
                raise ValueError("computed concentration-form equilibrium constant must be positive and finite")

            std_ratio = float(std_prod / std_react)

            def _fmt_side(sto: Dict[str, float]) -> str:
                out: List[str] = []
                for name in sorted(sto.keys()):
                    coeff = float(sto[name])
                    if abs(coeff - 1.0) < 1e-12:
                        out.append(f"{name}")
                    else:
                        out.append(f"{_fmt_coeff(coeff)}{name}")
                return " + ".join(out)

            cm_id = f"feq__{_canonical_side(rxn.reactants)}__{_canonical_side(rxn.products)}"
            dG_eq_out = float(out_units.from_jmol(float(dG_eq)))
            equilibrium_lines.append(
                "equilibrium: "
                + f"{_fmt_side(rxn.reactants)} <-> {_fmt_side(rxn.products)}"
                + f"; kf={kf:.17g}; dG_eq={dG_eq_out:.12g}; cm_id={cm_id}; cm_std_ratio={std_ratio:.17g}"
            )

    # Build state definitions for TS channels (if any).
    for gs_name, stoich in gs_sides.items():
        state_kind[gs_name] = "GS"
        state_members[gs_name] = dict(stoich)
        state_energy_J[gs_name] = _sum_energy(stoich, species_G_std_J)
        state_std_prod[gs_name] = _std_product(stoich, species_std_M)
        state_deg[gs_name] = 1.0

    for ts_state, ts_species in ts_states.items():
        state_kind[ts_state] = "TS"
        state_energy_J[ts_state] = float(species_G_std_J[ts_species])
        state_std_prod[ts_state] = float(species_std_M[ts_species])
        state_deg[ts_state] = float(species_deg[ts_species] if species_deg[ts_species] else 1.0)

    # For readability, shift energies so the minimum GS state is ~0.
    shift = 0.0
    gs_energies = [state_energy_J[n] for n, k in state_kind.items() if k == "GS"]
    if gs_energies:
        shift = float(min(gs_energies))

    def _members_expr(sto: Dict[str, int]) -> str:
        return _canonical_side(sto)

    # Compose generated DSL body.
    generated_lines: List[str] = []
    generated_lines.append("# NOTE: This block is auto-generated. Do not edit by hand.")
    generated_lines.append(f"energy={out_energy_unit}")
    generated_lines.append(f"T={float(T):.12g}")
    generated_lines.append("")

    # Fast equilibria first (human-readable) then state network.
    if equilibrium_lines:
        generated_lines.append("# Fast equilibria (no TS): kf + dG_eq + cm_std_ratio authority")
        generated_lines.extend(equilibrium_lines)
        generated_lines.append("")

    if state_kind:
        generated_lines.append("# State network (TS channels): ΔG sliders control barriers and reaction free energies")
        for name in sorted(state_kind.keys()):
            kind = state_kind[name]
            energy_out = float(out_units.from_jmol(float(state_energy_J[name] - shift)))
            deg = state_deg[name]
            std_prod = state_std_prod[name]
            members = state_members.get(name)
            extras: List[str] = []
            extras.append(f"kind={kind}")
            extras.append(f"energy={energy_out:.12g}")
            extras.append(f"degeneracy={deg:.12g}")
            extras.append(f"std={std_prod:.12g}")
            if members is not None:
                extras.append(f"members={_members_expr(members)}")
            generated_lines.append(f"state: {name}, " + ", ".join(extras))
        for a, b in edges:
            generated_lines.append(f"edge: {a},{b}")

    body = "\n".join(generated_lines).rstrip() + "\n"
    return CompiledComp(spec=spec, species_G_std_J_per_mol=dict(species_G_std_J), generated_reaction_dsl=body)
