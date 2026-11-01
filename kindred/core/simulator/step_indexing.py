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

from typing import Dict, Iterator, List, Tuple

from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.mechanism_metadata import MechanismMetadataKeys

__all__ = [
    "get_step_index_map",
    "iter_canonical_parameters",
]

def get_step_index_map(mechanism: object) -> List[Dict[str, object]]:
    """Return the builder-owned step map; a missing map is a producer error."""
    return list(mechanism.metadata[MechanismMetadataKeys.STEP_INDEX_MAP])


def iter_canonical_parameters(mechanism: object) -> Iterator[Tuple[str, Dict[str, object], str]]:
    """
    Yield (param_name, step_entry, role).

    role is one of: "k", "kf", "kr", "Keq".
    """
    step_map = get_step_index_map(mechanism)
    namespace = build_namespace_from_mechanism(mechanism)
    for item in namespace.ordered_items:
        entry = step_map[item.source_index]
        yield (item.canonical_name, entry, item.info.role)
