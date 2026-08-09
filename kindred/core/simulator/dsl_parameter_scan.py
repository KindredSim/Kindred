"""
Parameter-analysis projection helpers for the simulator DSL.

This module owns the ``DslParameterAnalysis`` value type and projects parameter
analyses from an already admitted mechanism.  The parser consumes that typed
value directly; it does not own or lazily import the scanner's DTO.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from kindred.core.equilibrium_rate_authority import (
    effective_equilibrium_forward_rate,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
    step_entry_authored_role_is_editable,
)
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.validation import try_parse_callable_finite_float, try_parse_int

from .parameter_namespace import (
    build_namespace_from_mechanism,
)
from .step_indexing import get_step_index_map


@dataclass(frozen=True, slots=True)
class DslParameterAnalysis:
    """One immutable parameter projection produced by the DSL scanner."""

    name: str
    value: float
    context: str
    source: str
    step_index: int | None = None
    editable: bool = True
    unit: str | None = None
    role: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("DSL parameter name must be nonempty text.")
        if type(self.value) not in {int, float}:
            raise TypeError("DSL parameter values must be exact numbers.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("DSL parameter values must be finite.")
        object.__setattr__(self, "value", value)
        for attr in ("context", "source"):
            if type(getattr(self, attr)) is not str:
                raise TypeError(
                    f"DSL parameter {attr} must be an exact string."
                )
        if self.step_index is not None and type(self.step_index) is not int:
            raise TypeError(
                "DSL parameter step index must be an exact int or None."
            )
        if type(self.editable) is not bool:
            raise TypeError("DSL parameter editable must be bool.")
        for attr in ("unit", "role"):
            raw = getattr(self, attr)
            if raw is not None and type(raw) is not str:
                raise TypeError(
                    f"DSL parameter {attr} must be a string or None."
                )
        for attr in ("minimum", "maximum"):
            raw = getattr(self, attr)
            if raw is None:
                continue
            if type(raw) not in {int, float}:
                raise TypeError(
                    f"DSL parameter {attr} must be an exact number."
                )
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise ValueError(f"DSL parameter {attr} must be finite.")
            object.__setattr__(self, attr, numeric)


def _finite_value(value: object) -> float | None:
    parsed, ok = try_parse_callable_finite_float(value)
    return float(parsed) if ok else None


def _mechanism_temperature(mechanism: object) -> float:
    meta = getattr(mechanism, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        value = _finite_value(meta.get(MechanismMetadataKeys.TEMPERATURE_K))
        if value is not None:
            return float(value)
    raise ValueError(
        "Admitted mechanism is missing its validated temperature_K authority."
    )


def extract_parameters_from_mechanism(
    mechanism: object,
) -> list[DslParameterAnalysis]:
    """Project scanner-owned parameter analyses from one admitted mechanism."""

    namespace = build_namespace_from_mechanism(mechanism)
    parameters: list[DslParameterAnalysis] = []
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
                DslParameterAnalysis(
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
            DslParameterAnalysis(
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

    return parameters


__all__ = ["DslParameterAnalysis", "extract_parameters_from_mechanism"]
