"""Shared low-level simulator normalization and stoichiometry utilities."""

from __future__ import annotations

from typing import Dict

__all__ = [
    "STANDARD_ENERGY_UNIT_MAP",
    "normalize_energy_unit",
]


STANDARD_ENERGY_UNIT_MAP: Dict[str, str] = {
    "j/mol": "J/mol",
    "kj/mol": "kJ/mol",
    "kcal/mol": "kcal/mol",
}


def normalize_energy_unit(
    value: object,
    *,
    default: str | None = None,
) -> str:
    canonical = None
    if isinstance(value, str):
        canonical = STANDARD_ENERGY_UNIT_MAP.get(value.strip().lower())
    if canonical is not None:
        return canonical
    if default is not None:
        return default
    raise ValueError(f"unsupported energy_unit {value!r}")
