"""
Canonical global step-index naming helpers.

Canonical rule:
- Every kinetic step in DSL order gets a 1-based index N.
- Reaction step -> readout kN
- Equilibrium step -> readouts kfN, krN, and KeqN

The step map separately records parser-admitted source coordinates.  Consumers
must use that source map to distinguish writable authored values from derived
indexed readouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterator, List, Optional, Tuple

from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.validation import try_parse_int

__all__ = [
    "get_step_index_map",
    "iter_canonical_parameters",
    "lookup_source_coordinate_target",
]

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
    step_map = get_step_index_map(mechanism)
    namespace = build_namespace_from_mechanism(mechanism)
    for item in namespace.ordered_items:
        if item.source_index is None:
            continue
        if not (0 <= int(item.source_index) < len(step_map)):
            continue
        entry = step_map[int(item.source_index)]
        role = item.info.role
        if role is None:
            continue
        yield (item.canonical_name, entry, role)


def lookup_source_coordinate_target(
    mechanism: object, name: str
) -> Optional[Tuple[str, int, str, Dict[str, object]]]:
    """Resolve one parser-admitted physical coordinate to its owning step.

    The source-coordinate map is populated by the parser/build seam and keeps
    physical model roles distinct from canonical indexed rate readouts.  This
    lookup therefore never infers ownership from a token spelling or from a
    derived mechanism value.
    """

    requested = str(name)
    namespace = build_namespace_from_mechanism(mechanism)
    if requested not in namespace.source_coordinate_names:
        return None
    for entry in get_step_index_map(mechanism):
        raw_coordinates = entry.get("source_coordinates") or {}
        if not isinstance(raw_coordinates, Mapping):
            continue
        for raw_role, raw_name in raw_coordinates.items():
            if str(raw_name) != requested:
                continue
            kind = str(entry.get("kind") or "")
            index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
            index, ok = try_parse_int(entry.get(index_key))
            if not ok or kind not in {"reaction", "equilibrium"}:
                return None
            return (kind, int(index), str(raw_role), entry)
    return None
