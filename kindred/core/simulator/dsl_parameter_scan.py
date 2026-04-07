"""
Parameter-scan helpers for the simulator DSL.

This module keeps reaction/algebra parameter name extraction out of the public
`dsl.py` entrypoint so the core parser facade stays smaller and import-thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import parameter_algebra
from .dsl import (
    _ARROW_RE,
    _extract_numeric_value,
    _parse_dsl_ir,
    _parse_stoich,
    _split_stoich_and_params,
)
from .dsl_format import format_stoichiometry_side as _fmt_side
from .errors import DSLError


@dataclass(frozen=True)
class ParameterDefinition:
    """
    Structured parameter entry extracted from the DSL.

    Attributes
    ----------
    name : str
        Parameter identifier (k, kf, kr, A, etc.).
    value : float
        Numeric value parsed from the DSL (unit-neutral).
    context : str
        Human-readable reaction context, e.g. "A + B -> C".
    source : str
        Additional source metadata (Arrhenius, Eyring, etc.).
    step_index : int | None
        1-based index of the reaction/equilibrium line in the DSL.
    """

    name: str
    value: float
    context: str
    source: str
    step_index: int | None = None


def _parameter_family(key: str) -> str | None:
    """
    Determine the canonical parameter family for a DSL key.

    Recognizes keys like k, k1, kf2, kr, A, Ea, dG_act, dG_eq, and Keq variants.
    """
    normalized = key.strip()
    lower = normalized.lower()

    families = ["kf", "kr", "k", "a", "ea", "dg_act", "dg_eq"]
    for fam in families:
        fam_lower = fam.lower()
        if lower == fam_lower:
            return fam
        suffix = lower[len(fam_lower):]
        if lower.startswith(fam_lower) and suffix.isdigit():
            return fam

    if normalized == "Keq":
        return "Keq"
    if normalized.startswith("Keq") and normalized[3:].isdigit():
        return "Keq"

    return None


def extract_parameters_from_dsl(text: str) -> list[ParameterDefinition]:
    """
    Extract explicit reaction parameters from DSL content.

    Supports both modern and shorthand reaction syntax:
    - Modern: reaction: A -> B; k=1.0
    - Shorthand: A -> B ; k=1.0 (used in preset files like M1)
    """
    parameters: list[ParameterDefinition] = []
    lines = [ln.rstrip() for ln in text.splitlines()]
    step_index = 0

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("comp:"):
            continue

        lower = line.lower()
        if lower.startswith(("reaction:", "equilibrium:")):
            step_index += 1
            _, rest = line.split(":", 1)
        elif _ARROW_RE.search(line):
            step_index += 1
            rest = line
        else:
            continue

        try:
            stoich_part, kv = _split_stoich_and_params(
                rest,
                reject_duplicate_canonical_keys=True,
            )
            react, prod, arrow = _parse_stoich(stoich_part)
        except DSLError as exc:
            if str(exc.message).startswith("Duplicate parameter:"):
                raise
            # Best-effort extraction utility: ignore malformed lines rather than failing the caller.
            continue

        if not kv:
            continue

        context = f"{_fmt_side(react)} {arrow} {_fmt_side(prod)}"

        for key, value_expr in kv.items():
            family = _parameter_family(key)
            if family is None:
                continue

            try:
                numeric_value = _extract_numeric_value(value_expr)
            except DSLError:
                numeric_value = None
            if numeric_value is None:
                continue

            if family in ("a", "ea"):
                source = "Arrhenius"
            elif family == "dg_act":
                source = "Eyring"
            elif family in ("dg_eq", "Keq"):
                source = "Equilibrium constant"
            elif family == "kf":
                source = "Forward rate"
            elif family == "kr":
                source = "Reverse rate"
            else:
                source = "Rate constant"

            parameters.append(
                ParameterDefinition(
                    name=key,
                    value=numeric_value,
                    context=context,
                    source=source,
                    step_index=step_index,
                )
            )

    return parameters


def extract_parameter_names_from_dsl(text: str) -> set[str]:
    """
    Extract all parameter names from DSL content.

    This function extracts parameter names from:
    1. Reaction parameter definitions (k, kf, kr, A, Ea, dG_act, etc.)
    2. Algebra section variable definitions in the `# Algebra` section:
       - `param name = ...` (solver/parameter-algebra)
       - `let name = ...` and `name = ...` (observables)
    """
    param_names: set[str] = set()

    reaction_params = extract_parameters_from_dsl(text)
    for param in reaction_params:
        param_names.add(param.name)

    ir = _parse_dsl_ir(text)
    n_rxn = sum(1 for step in ir.steps if not step.is_equilibrium)
    n_eq = sum(1 for step in ir.steps if step.is_equilibrium)

    mechanism_param_names: set[str] = {f"k{i}" for i in range(1, n_rxn + 1)}
    for i in range(1, n_eq + 1):
        mechanism_param_names.add(f"kf{i}")
        mechanism_param_names.add(f"kr{i}")
        mechanism_param_names.add(f"Keq{i}")

    spec = parameter_algebra.parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_param_names=mechanism_param_names,
    )
    param_names.update(spec.observable_names)
    param_names.update({assignment.name for assignment in spec.param_statements})

    return param_names
