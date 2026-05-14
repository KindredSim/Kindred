"""
GUI-facing enumeration of canonical solver parameters.

This is intentionally Qt-free so it can be unit-tested without launching a GUI.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
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


def _equilibrium_keq_value(eq: object) -> tuple[float, bool]:
    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, dict):
        value, ok = _try_finite_float(meta.get(EquilibriumMetadataKeys.KEQ_INPUT))
        if ok:
            return value, True
    kf, kf_ok = _try_finite_float(getattr(eq, "kf", None))
    kr, kr_ok = _try_finite_float(getattr(eq, "kr", None))
    if kf_ok and kr_ok and kr != 0.0:
        return float(kf) / float(kr), True
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
            has_K = bool(entry.get("has_Keq_param"))
            derive_rate = str(entry.get("derive_rate") or "")

            name = item.canonical_name
            if role == "kf":
                value, ok = _try_finite_float(getattr(eq, "kf", None))
            elif role == "kr":
                value, ok = _try_finite_float(getattr(eq, "kr", None))
            elif role == "Keq":
                value, ok = _equilibrium_keq_value(eq)
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
            if has_K:
                # Mark one rate as derived so Keq always has semantics.
                if role == derive_rate:
                    metadata[name]["editable"] = False
                    metadata[name]["derived"] = True
            elif role == "Keq":
                metadata[name]["editable"] = False
                metadata[name]["derived"] = True

    return variables, metadata
