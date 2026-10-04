"""
Builtins and protected symbol table for the Algebra model.

Current contract
----------------
- Built-in functions:
    sqrt, ln, log10, log1p, exp, expm1, sin, cos, tan, abs, min, max, pow, erf
- Helpers:
    heaviside(x) with heaviside(0)=0.0, clip(x, lo, hi), ifelse(cond, a, b)
- Protected read-only symbols:
    k_B, h, h_bar, N_A, R, T
- No shadowing of species or builtins; parser/evaluator must emit E120.

Notes
-----
- This module performs no I/O and keeps zero global mutable state.
- `T` exists in the table and is protected against user redefinition.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Set

from ..constants import k_B, h, h_bar, N_A, R

logger = logging.getLogger(__name__)


__all__ = [
    "heaviside",
    "clip",
    "ifelse",
    "BUILTIN_FUNCTIONS",
    "PROTECTED_NAMES",
    "SymbolTable",
]


# ----------------------------- helper functions ------------------------------

def heaviside(x: float) -> float:
    """
    Heaviside step with H(0)=0.0 as per spec.
    Returns 0.0 for x <= 0, 1.0 for x > 0.
    """
    try:
        xf = float(x)
    except (TypeError, ValueError) as exc:
        logger.debug("Failed to convert value to float in heaviside: %r", x, exc_info=True)
        raise TypeError(f"heaviside expects a numeric argument, got {x!r}") from exc
    if xf <= 0.0:
        return 0.0
    return 1.0


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    xf = float(x)
    lof = float(lo)
    hif = float(hi)
    if lof > hif:
        # Swap to be conservative and deterministic
        lof, hif = hif, lof
    return min(max(xf, lof), hif)


def ifelse(cond: Any, a: float, b: float) -> float:
    """
    Return a if cond is truthy, else b.

    The evaluator is responsible for raising E170 where needed; this helper
    obeys normal Python truthiness.
    """
    return float(a) if bool(cond) else float(b)


# ----------------------------- builtins catalog ------------------------------

@dataclass(frozen=True)
class BuiltinFunction:
    evaluate: Callable[..., float]
    unit_rule: str

    def __call__(self, *args):
        return self.evaluate(*args)


BUILTIN_FUNCTIONS: Dict[str, BuiltinFunction] = {
    "sqrt": BuiltinFunction(lambda x: math.sqrt(float(x)), "sqrt"),
    "ln": BuiltinFunction(lambda x: math.log(float(x)), "dimensionless"),
    "log10": BuiltinFunction(lambda x: math.log10(float(x)), "dimensionless"),
    "log1p": BuiltinFunction(lambda x: math.log1p(float(x)), "dimensionless"),
    "exp": BuiltinFunction(lambda x: math.exp(float(x)), "dimensionless"),
    "expm1": BuiltinFunction(lambda x: math.expm1(float(x)), "dimensionless"),
    "sin": BuiltinFunction(lambda x: math.sin(float(x)), "dimensionless"),
    "cos": BuiltinFunction(lambda x: math.cos(float(x)), "dimensionless"),
    "tan": BuiltinFunction(lambda x: math.tan(float(x)), "dimensionless"),
    "abs": BuiltinFunction(lambda x: abs(float(x)), "preserve"),
    "min": BuiltinFunction(lambda *xs: float(min(float(v) for v in xs)), "compatible"),
    "max": BuiltinFunction(lambda *xs: float(max(float(v) for v in xs)), "compatible"),
    "pow": BuiltinFunction(lambda x, y: float(math.pow(float(x), float(y))), "power"),
    "erf": BuiltinFunction(lambda x: math.erf(float(x)), "dimensionless"),
    "heaviside": BuiltinFunction(heaviside, "predicate"),
    "clip": BuiltinFunction(clip, "compatible"),
    "ifelse": BuiltinFunction(ifelse, "branches"),
}

# Protected names are read-only within Algebra
PROTECTED_NAMES: Set[str] = {"k_B", "h", "h_bar", "N_A", "R", "T"}



# ----------------------------- symbol table ----------------------------------

@dataclass
class SymbolTable:
    """
    Symbol table exposing protected builtins.

    Behavior
    --------
    - Protected names (constants and T) cannot be shadowed by user code.
    - Built-in function names cannot be used as symbols.
    """

    _values: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize protected constants and T
        self._values = {
            "k_B": float(k_B),
            "h": float(h),
            "h_bar": float(h_bar),
            "N_A": float(N_A),
            "R": float(R),
            "T": 298.15,
        }

    # --- lookup --------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Return True if a protected symbol exists."""
        return name in self._values

    def get(self, name: str) -> float:
        """Get a symbol value, raising KeyError if unknown."""
        return self._values[name]

    # --- function exposure ---------------------------------------------------

    def functions(self) -> Mapping[str, Callable[..., float]]:
        """Return read-only view of built-in functions."""
        return BUILTIN_FUNCTIONS

    # --- discovery -----------------------------------------------------------

    def protected_names(self) -> Set[str]:
        """Return the set of protected symbol names."""
        return set(PROTECTED_NAMES)
