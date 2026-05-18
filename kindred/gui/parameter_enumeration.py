"""
GUI-facing enumeration of canonical solver parameters.

This is intentionally Qt-free so it can be unit-tested without launching a GUI.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Tuple

from kindred.core.equilibrium_rate_authority import (
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
    step_entry_role_derived,
    step_entry_role_editable,
)
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.validation import try_parse_finite_float, try_parse_int

__all__ = ["enumerate_step_parameters_for_gui"]


def _try_finite_float(x: object) -> tuple[float, bool]:
    try:
        raw = x() if callable(x) else x
    except Exception:
        return 0.0, False
    return try_parse_finite_float(raw)


def _mechanism_temperature(mechanism: object) -> float:
    meta = getattr(mechanism, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        value, ok = _try_finite_float(meta.get(MechanismMetadataKeys.TEMPERATURE_K))
        if ok:
            return float(value)
    return 298.15


def _equilibrium_keq_value(eq: object, *, temperature_K: float) -> tuple[float, bool]:
    value = effective_equilibrium_keq(eq, temperature_K=float(temperature_K))
    if value is not None:
        return float(value), True
    return 0.0, False


def enumerate_step_parameters_for_gui(mechanism: object) -> Tuple["OrderedDict[str, float]", Dict[str, Dict[str, object]]]:
    """
    Return (variables, metadata) for canonical step-index parameters only.

    - variables maps canonical parameter name -> current numeric value
    - metadata includes:
        type: 'reaction' or 'equilibrium'
        index: canonical step index (1-based)
        role: 'k'/'kf'/'kr'/'Keq'
        label: 'Step N: <context>'
        editable/derived for Keq-implied constraints (kr or kf derived when Keq is explicit)
    """
    variables: "OrderedDict[str, float]" = OrderedDict()
    metadata: Dict[str, Dict[str, object]] = {}

    step_map = get_step_index_map(mechanism)
    namespace = build_namespace_from_mechanism(mechanism)
    step_entry_by_index = {}
    for entry in step_map:
        n, ok = try_parse_int(entry.get("step_index"))
        if ok:
            step_entry_by_index[int(n)] = entry
    rxns = list(getattr(mechanism, "reactions", []) or [])
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    temperature_K = _mechanism_temperature(mechanism)
    for item in namespace.ordered_items:
        info = item.info
        n = info.step_index
        if n is None:
            continue
        entry = step_entry_by_index.get(int(n), {})
        kind = str(info.step_kind or "")
        role = str(info.role or "")
        context = str(entry.get("context") or "")
        label = f"Step {n}: {context}".strip()
        if kind == "reaction":
            idx, ok = try_parse_int(entry.get("reaction_index", -1))
            if not ok:
                continue
            if not (0 <= idx < len(rxns)):
                continue
            name = item.canonical_name
            value, ok = _try_finite_float(getattr(rxns[idx], "rate", None))
            variables[name] = float(value)
            metadata[name] = {"type": "reaction", "index": n, "role": "k", "label": label, "value_valid": bool(ok)}
        elif kind == "equilibrium":
            idx, ok = try_parse_int(entry.get("equilibrium_index", -1))
            if not ok:
                continue
            if not (0 <= idx < len(eqs)):
                continue
            eq = eqs[idx]
            authority = normalize_existing_equilibrium_rate_authority(eq)

            name = item.canonical_name
            if role == "kf":
                value, ok = _try_finite_float(getattr(eq, "kf", None))
            elif role == "kr":
                effective_kr = effective_equilibrium_reverse_rate(eq, temperature_K=temperature_K)
                value, ok = (float(effective_kr), True) if effective_kr is not None else (0.0, False)
            elif role == "Keq":
                value, ok = _equilibrium_keq_value(eq, temperature_K=temperature_K)
            else:
                continue
            variables[name] = float(value)
            metadata[name] = {
                "type": "equilibrium",
                "index": n,
                "role": role,
                "label": label,
                "value_valid": bool(ok),
            }
            editable = step_entry_role_editable(entry, role)
            derived = step_entry_role_derived(entry, role)
            if editable is None:
                editable = authority.role_editability(role)
            if derived is None:
                derived = authority.role_derived(role)
            if not editable:
                metadata[name]["editable"] = False
            if derived:
                metadata[name]["derived"] = True

    return variables, metadata
