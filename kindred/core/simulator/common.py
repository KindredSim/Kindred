"""Shared low-level simulator normalization and stoichiometry utilities."""

from __future__ import annotations

import math
from typing import Dict
from .errors import non_positive_stoichiometry_error

__all__ = [
    "STANDARD_ENERGY_UNIT_MAP",
    "molecularity",
    "normalize_energy_unit",
]


STANDARD_ENERGY_UNIT_MAP: Dict[str, str] = {
    "j/mol": "J/mol",
    "kj/mol": "kJ/mol",
    "kcal/mol": "kcal/mol",
    "hartree": "hartree",
}


def normalize_energy_unit(
    value: object,
    *,
    default: str | None = None,
    allow_hartree: bool = False,
) -> str:
    canonical = None
    if isinstance(value, str):
        canonical = STANDARD_ENERGY_UNIT_MAP.get(value.strip().lower())
        if canonical == "hartree" and not allow_hartree:
            canonical = None
    if canonical is not None:
        return canonical
    if default is not None:
        return default
    raise ValueError(f"unsupported energy_unit {value!r}")


def molecularity(reactants: Dict[str, float]) -> float:
    coefficients: list[float] = []
    for v in (reactants or {}).values():
        if type(v) is bool:
            raise non_positive_stoichiometry_error()
        try:
            coefficient = float(v)
        except (TypeError, ValueError, OverflowError) as exc:
            raise non_positive_stoichiometry_error() from exc
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise non_positive_stoichiometry_error()
        coefficients.append(coefficient)
    total = math.fsum(coefficients)
    if not math.isfinite(total):
        raise non_positive_stoichiometry_error()
    return float(total)
