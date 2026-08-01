"""
Parameter-scan helpers for the simulator DSL.

This module keeps reaction/algebra parameter name extraction out of the public
`dsl.py` entrypoint so the core parser facade stays smaller and import-thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kindred.core.equilibrium_rate_authority import (
    effective_equilibrium_forward_rate,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    generated_fast_equilibrium_runtime_parameter_name,
    normalize_existing_equilibrium_rate_authority,
    step_entry_authored_role_is_editable,
)
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataKeys
from kindred.core.units import UnitsModel
from kindred.core.validation import try_parse_callable_finite_float, try_parse_int

from .parameter_namespace import (
    build_namespace_from_mechanism,
)
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
    unit : str | None
        Unit owned by the source endpoint, when the value is unitful.
    role : str | None
        Source role for structured endpoints such as generated fast equilibria.
    min/max : float | None
        Optional source-owned default scan range.
    """

    name: str
    value: float
    context: str
    source: str
    step_index: int | None = None
    editable: bool = True
    unit: str | None = None
    role: str | None = None
    min: float | None = None
    max: float | None = None


def _finite_value(value: object) -> float | None:
    parsed, ok = try_parse_callable_finite_float(value)
    return float(parsed) if ok else None


def _mechanism_temperature(mechanism: object) -> float:
    meta = getattr(mechanism, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        value = _finite_value(meta.get(MechanismMetadataKeys.TEMPERATURE_K))
        if value is not None:
            return float(value)
    return 298.15


def extract_parameters_from_mechanism(
    mechanism: object,
) -> list[ParameterDefinition]:
    """Project parameters from one already admitted mechanism."""

    namespace = build_namespace_from_mechanism(mechanism)
    parameters: list[ParameterDefinition] = []
    rxns = list(getattr(mechanism, "reactions", []) or [])
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    temperature_K = _mechanism_temperature(mechanism)
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
        authority = normalize_existing_equilibrium_rate_authority(eq)
        kf_value = effective_equilibrium_forward_rate(eq, temperature_K=temperature_K)
        kr_value = effective_equilibrium_reverse_rate(eq, temperature_K=temperature_K)
        source, value = {
            "kf": ("Forward rate", kf_value),
            "kr": ("Reverse rate", kr_value),
            "Keq": ("Equilibrium constant", effective_equilibrium_keq(eq, temperature_K=temperature_K)),
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
                editable=bool(
                    step_entry_authored_role_is_editable(entry, role)
                    if step_entry_authored_role_is_editable(entry, role) is not None
                    else authority.authored_role_is_editable(role)
                ),
            )
        )

    step_entry_by_equilibrium_index: dict[int, dict] = {}
    for entry in step_entry_by_index.values():
        if not isinstance(entry, dict) or str(entry.get("kind") or "") != "equilibrium":
            continue
        equilibrium_index, ok = try_parse_int(entry.get("equilibrium_index", -1))
        if ok:
            step_entry_by_equilibrium_index[int(equilibrium_index)] = entry
    for equilibrium_index, eq in enumerate(eqs):
        metadata = dict(getattr(eq, "metadata", {}) or {})
        name = generated_fast_equilibrium_runtime_parameter_name(eq)
        if name is None:
            continue
        dG_eq_j = _finite_value(metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL))
        if dG_eq_j is None:
            continue
        source_energy_unit = str(
            metadata.get(EquilibriumMetadataKeys.SOURCE_ENERGY_UNIT)
            or (getattr(mechanism, "metadata", {}) or {}).get(MechanismMetadataKeys.ENERGY_UNIT)
            or "kJ/mol"
        )
        unit_conv = UnitsModel(energy_unit=source_energy_unit)
        source_value = float(unit_conv.from_jmol(float(dG_eq_j)))
        span = max(10.0, abs(source_value) * 0.5)
        entry = step_entry_by_equilibrium_index.get(int(equilibrium_index), {})
        step_index = None
        parsed_step_index, step_ok = try_parse_int(entry.get("step_index")) if entry else (None, False)
        if step_ok:
            step_index = int(parsed_step_index)
        context = str(entry.get("context") or metadata.get("cm_id") or name)
        parameters.append(
            ParameterDefinition(
                name=name,
                value=source_value,
                context=context,
                source="Generated fast dG_eq",
                step_index=step_index,
                editable=True,
                unit=source_energy_unit,
                role="dG_eq_fast",
                min=float(source_value - span),
                max=float(source_value + span),
            )
        )

    return parameters


__all__ = ["ParameterDefinition", "extract_parameters_from_mechanism"]
