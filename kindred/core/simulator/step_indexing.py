"""
Canonical global step-index naming helpers.

Canonical rule:
- Every kinetic step in DSL order gets a 1-based index N.
- Reaction step -> parameter kN
- Equilibrium step -> parameters kfN and krN
- Equilibrium constant parameter KeqN exists only when explicitly represented/used
  (currently: when the DSL provided K=... on that step; stored as metadata Keq_input).

Policy:
- State-network generated steps are excluded from the step-index map to avoid
  collisions with user-visible names (see parse_dsl_to_mechanism).
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Set, Tuple

from kindred.core.validation import try_parse_int

__all__ = [
    "canonical_parameter_names",
    "get_step_index_map",
    "iter_canonical_parameters",
    "lookup_step_param_target",
]

_CANON_PARAM_RE = re.compile(r"^(k|kf|kr|Keq)(\d+)$")


def get_step_index_map(mechanism: object) -> List[Dict[str, object]]:
    meta = getattr(mechanism, "metadata", {}) or {}
    mapping = meta.get("step_index_map")
    if not isinstance(mapping, list):
        return []
    out: List[Dict[str, object]] = []
    for entry in mapping:
        if not isinstance(entry, dict):
            continue
        if "step_index" not in entry or "kind" not in entry:
            continue
        out.append(entry)
    return out


def iter_canonical_parameters(mechanism: object) -> Iterator[Tuple[str, Dict[str, object], str]]:
    """
    Yield (param_name, step_entry, role).

    role is one of: "k", "kf", "kr", "Keq".
    """
    for entry in get_step_index_map(mechanism):
        n, ok = try_parse_int(entry.get("step_index"))
        if not ok:
            continue
        kind = str(entry.get("kind") or "")
        if kind == "reaction":
            yield (f"k{n}", entry, "k")
        elif kind == "equilibrium":
            yield (f"kf{n}", entry, "kf")
            yield (f"kr{n}", entry, "kr")
            if bool(entry.get("has_Keq_param")):
                yield (f"Keq{n}", entry, "Keq")


def canonical_parameter_names(mechanism: object) -> Set[str]:
    return {name for name, _entry, _role in iter_canonical_parameters(mechanism)}


def lookup_step_param_target(
    mechanism: object, name: str
) -> Optional[Tuple[str, int, str, Dict[str, object]]]:
    """
    Resolve a canonical parameter name to a mechanism target.

    Returns (kind, index, role, entry) where:
    - kind: "reaction" or "equilibrium"
    - index: 0-based index into mechanism.reactions or mechanism.equilibria
    - role: "k" / "kf" / "kr" / "Keq"
    """
    m = _CANON_PARAM_RE.match(str(name))
    if not m:
        return None
    role = m.group(1)
    step_no, ok = try_parse_int(m.group(2))
    if not ok:
        return None
    for entry in get_step_index_map(mechanism):
        n, ok = try_parse_int(entry.get("step_index"))
        if not ok:
            continue
        if n != step_no:
            continue
        kind = str(entry.get("kind") or "")
        if role == "k":
            if kind != "reaction":
                return None
            idx_raw = entry.get("reaction_index")
            idx, ok = try_parse_int(idx_raw)
            if not ok:
                return None
            return ("reaction", idx, "k", entry)
        if role in {"kf", "kr", "Keq"}:
            if kind != "equilibrium":
                return None
            if role == "Keq" and not bool(entry.get("has_Keq_param")):
                return None
            idx_raw = entry.get("equilibrium_index")
            idx, ok = try_parse_int(idx_raw)
            if not ok:
                return None
            return ("equilibrium", idx, role, entry)
    return None
