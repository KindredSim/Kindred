"""
GUI-facing enumeration of canonical solver parameters.

This is intentionally Qt-free so it can be unit-tested without launching a GUI.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.validation import try_parse_finite_float, try_parse_int

__all__ = ["enumerate_step_parameters_for_gui"]


def _try_finite_float(x: object) -> tuple[float, bool]:
    try:
        raw = x() if callable(x) else x
    except Exception:
        return 0.0, False
    return try_parse_finite_float(raw)


def enumerate_step_parameters_for_gui(mechanism: object) -> Tuple["OrderedDict[str, float]", Dict[str, Dict[str, object]]]:
    """
    Return (variables, metadata) for canonical step-index parameters only.

    - variables maps canonical parameter name -> current numeric value
    - metadata includes:
        type: 'reaction' or 'equilibrium'
        index: canonical step index (1-based)
        role: 'k'/'kf'/'kr'/'K'
        label: 'Step N: <context>'
        editable/derived for K-implied constraints (kr or kf derived when K is explicit)
    """
    variables: "OrderedDict[str, float]" = OrderedDict()
    metadata: Dict[str, Dict[str, object]] = {}

    step_map = get_step_index_map(mechanism)
    rxns = list(getattr(mechanism, "reactions", []) or [])
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    for entry in step_map:
        kind = str(entry.get("kind") or "")
        n, ok = try_parse_int(entry.get("step_index"))
        if not ok:
            continue
        context = str(entry.get("context") or "")
        label = f"Step {n}: {context}".strip()
        if kind == "reaction":
            idx, ok = try_parse_int(entry.get("reaction_index", -1))
            if not ok:
                continue
            if not (0 <= idx < len(rxns)):
                continue
            name = f"k{n}"
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
            has_K = bool(entry.get("has_K_param"))
            derive_rate = str(entry.get("derive_rate") or "")

            kf_name = f"kf{n}"
            kr_name = f"kr{n}"
            kf_value, kf_ok = _try_finite_float(getattr(eq, "kf", None))
            kr_value, kr_ok = _try_finite_float(getattr(eq, "kr", None))
            variables[kf_name] = float(kf_value)
            variables[kr_name] = float(kr_value)
            metadata[kf_name] = {
                "type": "equilibrium",
                "index": n,
                "role": "kf",
                "label": label,
                "value_valid": bool(kf_ok),
            }
            metadata[kr_name] = {
                "type": "equilibrium",
                "index": n,
                "role": "kr",
                "label": label,
                "value_valid": bool(kr_ok),
            }

            if has_K:
                K_name = f"K{n}"
                meta = getattr(eq, "metadata", {}) or {}
                k_value, k_ok = _try_finite_float(meta.get("K_input"))
                variables[K_name] = float(k_value)
                metadata[K_name] = {
                    "type": "equilibrium",
                    "index": n,
                    "role": "K",
                    "label": label,
                    "value_valid": bool(k_ok),
                }

                # Mark one rate as derived so K always has semantics.
                if derive_rate == "kf":
                    metadata[kf_name]["editable"] = False
                    metadata[kf_name]["derived"] = True
                else:
                    metadata[kr_name]["editable"] = False
                    metadata[kr_name]["derived"] = True

    return variables, metadata
