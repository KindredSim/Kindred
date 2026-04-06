"""
Shared simulator utilities used across DSL parsing and fast-equilibrium policy code.

Architecture note: `dsl.py` and `fast_eq.py` must not import each other.
This module is the stable shared layer that both can depend on without creating cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .errors import DSLError, non_integer_molecularity_error, non_positive_stoichiometry_error
from .kinetics import K_from_deltaG_eq

__all__ = [
    "DSLError",
    "STANDARD_ENERGY_UNIT_MAP",
    "choose_k_fast",
    "derive_equilibrium_rates",
    "FastEqResult",
    "molecularity",
    "normalize_energy_unit",
]


STANDARD_ENERGY_UNIT_MAP: Dict[str, str] = {
    "j/mol": "J/mol",
    "j": "J/mol",
    "kj/mol": "kJ/mol",
    "kj": "kJ/mol",
    "kcal/mol": "kcal/mol",
    "kcal": "kcal/mol",
    "hartree": "hartree",
    "eh": "hartree",
}


_DIRECTIVE_ENERGY_UNIT_MAP: Dict[str, str] = {
    key: value
    for key, value in STANDARD_ENERGY_UNIT_MAP.items()
    if "/" in key and value != "hartree"
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


def _normalize_directive_energy_unit(value: object) -> str:
    if isinstance(value, str):
        canonical = _DIRECTIVE_ENERGY_UNIT_MAP.get(value.strip().lower())
        if canonical is not None:
            return canonical
    raise ValueError(f"unsupported energy_unit {value!r}")


def _pos_finite(x: float | None, label: str = "value") -> float:
    if x is None:
        raise DSLError(f"missing value for {label}")
    xf = float(x)
    if not (xf > 0.0) or not math.isfinite(xf):
        raise DSLError(f"{label} must be positive and finite")
    return xf


def molecularity(reactants: Dict[str, float]) -> int:
    tot = 0.0
    for v in (reactants or {}).values():
        if v <= 0:
            raise non_positive_stoichiometry_error()
        tot += float(v)
    n = int(round(tot))
    if abs(tot - n) > 1e-9:
        raise non_integer_molecularity_error()
    return max(1, n)


def choose_k_fast(
    explicit_rates: Optional[Iterable[float]] = None,
    *,
    fallback: float = 1e6,
    clamp_min: float = 1e3,
    clamp_max: float = 1e12,
) -> float:
    """
    Compute k_fast according to the policy:

        k_fast = clamp( 10 × max(explicit_rates), [clamp_min, clamp_max] )
        if explicit_rates empty or None → use fallback before clamping
    """
    lo = _pos_finite(clamp_min, "clamp_min")
    hi = _pos_finite(clamp_max, "clamp_max")
    if hi < lo:
        lo, hi = hi, lo
    base: float
    if explicit_rates:
        clean: List[float] = [
            float(k)
            for k in explicit_rates
            if isinstance(k, (int, float)) and math.isfinite(float(k)) and float(k) > 0.0
        ]
        if clean:
            base = 10.0 * max(clean)
        else:
            base = _pos_finite(fallback, "fallback")
    else:
        base = _pos_finite(fallback, "fallback")
    if base < lo:
        return lo
    if base > hi:
        return hi
    return base


@dataclass(frozen=True)
class FastEqResult:
    kf: float
    kr: float
    K: float
    k_fast: float
    source: str


def derive_equilibrium_rates(
    *,
    K: float | None = None,
    dG_eq_J_per_mol: float | None = None,
    T: float = 298.15,
    explicit_rates: Optional[Iterable[float]] = None,
) -> FastEqResult:
    """
    Derive (kf, kr) for an equilibrium step using the fast-equilibrium policy.

    You must provide either K or ΔG° (in J/mol). If both are provided, K takes precedence.
    """
    if K is None and dG_eq_J_per_mol is None:
        raise DSLError("derive_equilibrium_rates requires K or dG_eq_J_per_mol")

    if K is None:
        if dG_eq_J_per_mol is None:
            raise DSLError("missing value for dG_eq_J_per_mol")
        dG_eq_val = float(dG_eq_J_per_mol)
        if not math.isfinite(dG_eq_val):
            raise DSLError("dG_eq_J_per_mol must be finite")
        K_val = K_from_deltaG_eq(dG_eq_val, _pos_finite(T, "T"))
        source = "derived(ΔG°)"
    else:
        K_val = _pos_finite(K, "K")
        source = "derived(K)"

    kf_fast = choose_k_fast(explicit_rates)
    kr_fast = kf_fast / K_val
    return FastEqResult(kf=kf_fast, kr=kr_fast, K=K_val, k_fast=kf_fast, source=source)
