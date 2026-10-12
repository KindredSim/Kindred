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

from typing import Dict, Iterator, List, Optional, Tuple

from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

__all__ = [
    "get_step_index_map",
    "iter_canonical_parameters",
    "lookup_source_coordinate_target",
]

def get_step_index_map(mechanism: object) -> List[Dict[str, object]]:
    """Return the builder-owned step map; a missing map is a producer error."""
    return list(mechanism.metadata["step_index_map"])


def iter_canonical_parameters(mechanism: object) -> Iterator[Tuple[str, Dict[str, object], str]]:
    """
    Yield (param_name, step_entry, role).

    role is one of: "k", "kf", "kr", "Keq".
    """
    step_map = get_step_index_map(mechanism)
    namespace = build_namespace_from_mechanism(mechanism)
    for item in namespace.ordered_items:
        entry = step_map[int(item.source_index)]
        yield (item.canonical_name, entry, item.info.role)


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
        for raw_role, raw_name in entry["source_coordinates"].items():
            if str(raw_name) != requested:
                continue
            kind = str(entry["kind"])
            index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
            return (kind, int(entry[index_key]), str(raw_role), entry)
    return None
