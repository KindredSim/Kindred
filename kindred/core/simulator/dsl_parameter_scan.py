"""
Parameter-scan helpers for the simulator DSL.

This module keeps reaction/algebra parameter name extraction out of the public
`dsl.py` entrypoint so the core parser facade stays smaller and import-thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from kindred.core.algebra.simulation_series import compile_algebra_observables
from . import parameter_algebra
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.validation import try_parse_callable_finite_float, try_parse_int

from .dsl import _parse_dsl_ir
from .dsl_build import build_mechanism_from_ir
from .errors import DSLError
from .parameter_algebra_spec import collect_algebra_section_lines
from .parameter_namespace import build_namespace_from_ir_steps, build_namespace_from_mechanism
from .step_indexing import get_step_index_map


@dataclass(frozen=True)
class ParameterDefinition:
    """
    Structured parameter entry extracted from the DSL.

    Attributes
    ----------
    name : str
        Canonical indexed mechanism parameter identifier (k1, kf2, kr2, Keq2).
    value : float
        Numeric value parsed from the DSL (unit-neutral).
    context : str
        Human-readable reaction context, e.g. "A + B -> C".
    source : str
        Additional source metadata (Arrhenius, Eyring, etc.).
    step_index : int | None
        1-based index of the reaction/equilibrium line in the DSL.
    editable : bool
        Whether the parameter is a writable public/fitting endpoint.
    """

    name: str
    value: float
    context: str
    source: str
    step_index: int | None = None
    editable: bool = True


def _finite_value(value: object) -> float | None:
    parsed, ok = try_parse_callable_finite_float(value)
    return float(parsed) if ok else None


def _equilibrium_keq_value(eq: object) -> float | None:
    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, dict):
        explicit_value = _finite_value(meta.get(EquilibriumMetadataKeys.KEQ_INPUT))
        if explicit_value is not None:
            return explicit_value
    kf = _finite_value(getattr(eq, "kf", None))
    kr = _finite_value(getattr(eq, "kr", None))
    if kf is None or kr in (None, 0.0):
        return None
    return float(kf) / float(kr)


def extract_parameters_from_dsl(text: str) -> list[ParameterDefinition]:
    """
    Extract explicit reaction parameters from DSL content.

    Supports both modern and shorthand reaction syntax:
    - Modern: reaction: A -> B; k=1.0
    - Shorthand: A -> B ; k=1.0 (used in preset files like M1)
    """
    ir = _parse_dsl_ir(text)
    mechanism = build_mechanism_from_ir(ir, initials={})
    namespace = build_namespace_from_mechanism(mechanism)
    parameters: list[ParameterDefinition] = []
    rxns = list(getattr(mechanism, "reactions", []) or [])
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    step_entry_by_index = {}
    for entry in get_step_index_map(mechanism):
        step_index, ok = try_parse_int(entry.get("step_index"))
        if ok:
            step_entry_by_index[int(step_index)] = entry

    for item in namespace.ordered_items:
        info = item.info
        step_index = info.step_index
        if step_index is None:
            continue
        entry = step_entry_by_index.get(int(step_index), {})
        kind = str(info.step_kind or "")
        role = str(info.role or "")
        context = str(entry.get("context") or "")
        if kind == "reaction":
            reaction_index, ok = try_parse_int(entry.get("reaction_index", -1))
            if not ok or not (0 <= reaction_index < len(rxns)):
                continue
            value = _finite_value(getattr(rxns[reaction_index], "rate", None))
            if value is None:
                continue
            parameters.append(
                ParameterDefinition(
                    name=item.canonical_name,
                    value=float(value),
                    context=context,
                    source="Rate constant",
                    step_index=int(step_index),
                )
            )
            continue
        if kind != "equilibrium":
            continue
        equilibrium_index, ok = try_parse_int(entry.get("equilibrium_index", -1))
        if not ok or not (0 <= equilibrium_index < len(eqs)):
            continue
        eq = eqs[equilibrium_index]
        has_explicit_keq = bool(entry.get("has_Keq_param"))
        derive_rate = str(entry.get("derive_rate") or "kr")
        source, value = {
            "kf": ("Forward rate", _finite_value(getattr(eq, "kf", None))),
            "kr": ("Reverse rate", _finite_value(getattr(eq, "kr", None))),
            "Keq": ("Equilibrium constant", _equilibrium_keq_value(eq)),
        }.get(role, ("", None))
        if value is None:
            continue
        parameters.append(
            ParameterDefinition(
                name=item.canonical_name,
                value=float(value),
                context=context,
                source=source,
                step_index=int(step_index),
                editable=bool((role != "Keq" or has_explicit_keq) and not (has_explicit_keq and role == derive_rate)),
            )
        )

    return parameters


def _scan_mechanism_param_names(ir) -> set[str]:
    return build_namespace_from_ir_steps(ir.steps).flat_names()


def extract_parameter_names_from_dsl(text: str) -> set[str]:
    """
    Extract all parameter names from DSL content.

    This function extracts parameter names from:
    1. Reaction parameter definitions (k, kf, kr, A, Ea, dG_act, etc.)
    2. Algebra declaration lines in the mechanism DSL:
       - `param name = ...` (solver/parameter-algebra)
       - `let name = ...` (observables)
    """
    param_names: set[str] = set()

    ir = _parse_dsl_ir(text)
    mechanism_namespace = build_namespace_from_ir_steps(ir.steps)
    param_names.update(mechanism_namespace.flat_names())

    spec = parameter_algebra.parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_namespace=mechanism_namespace,
    )
    algebra_lines = collect_algebra_section_lines(text)
    if algebra_lines:
        try:
            compile_algebra_observables(
                "\n".join(str(raw) for _line_no, raw in algebra_lines),
                mechanism_namespace=mechanism_namespace,
            )
        except ValueError as exc:
            raise DSLError(str(exc)) from exc
    param_names.update(spec.observable_names)
    param_names.update({assignment.name for assignment in spec.param_statements})

    return param_names
