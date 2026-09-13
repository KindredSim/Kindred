"""Selected scientific symbols shared by execution and result Algebra."""
from __future__ import annotations

def runtime_algebra_symbol_names(parameter_algebra_spec: object) -> frozenset[str]:
    """Classify row dependence without materializing parameter values."""
    indexed = {
        str(item.canonical_name)
        for item in parameter_algebra_spec.mechanism_namespace.ordered_items
        if item.info.role in {"k", "kf", "kr", "Keq"}
    }
    return frozenset(indexed | parameter_algebra_spec.temperature_dependent_names)
