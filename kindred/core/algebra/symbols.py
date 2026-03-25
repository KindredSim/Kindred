"""
Builtins and protected symbol table for the Algebra model.

Current contract
----------------
- Built-in functions:
    sqrt, ln, log10, log1p, exp, expm1, sin, cos, tan, abs, min, max, pow, erf
- Helpers:
    heaviside(x) with heaviside(0)=0.0, clip(x, lo, hi), ifelse(cond, a, b)
- Protected read-only symbols:
    k_B, h, ħ, N_A, R, Rkcal, T
- No shadowing of species or builtins; parser/evaluator must emit E120.

Notes
-----
- This module performs no I/O and keeps zero global mutable state.
- `T` exists in the table and is protected against user redefinition. The
  application or simulator may update it through `SymbolTable.update_temperature`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Set

from ..constants import k_B, h, hbar as ħ, N_A, R, Rkcal

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

BUILTIN_FUNCTIONS: Dict[str, Callable[..., float]] = {
    # Math core
    "sqrt": lambda x: math.sqrt(float(x)),
    "ln": lambda x: math.log(float(x)),
    "log10": lambda x: math.log10(float(x)),
    "log1p": lambda x: math.log1p(float(x)),
    "exp": lambda x: math.exp(float(x)),
    "expm1": lambda x: math.expm1(float(x)),
    "sin": lambda x: math.sin(float(x)),
    "cos": lambda x: math.cos(float(x)),
    "tan": lambda x: math.tan(float(x)),
    "abs": lambda x: abs(float(x)),
    "min": lambda *xs: float(min(float(v) for v in xs)),
    "max": lambda *xs: float(max(float(v) for v in xs)),
    "pow": lambda x, y: float(math.pow(float(x), float(y))),
    "erf": lambda x: math.erf(float(x)),
    # Helpers
    "heaviside": heaviside,
    "clip": clip,
    "ifelse": ifelse,
}

# Protected names are read-only within Algebra
PROTECTED_NAMES: Set[str] = {"k_B", "h", "ħ", "N_A", "R", "Rkcal", "T"}

# Names that cannot be defined by users either (functions)
PROTECTED_FUNCTION_NAMES: Set[str] = set(BUILTIN_FUNCTIONS.keys())


# ----------------------------- symbol table ----------------------------------

@dataclass
class SymbolTable:
    """
    Symbol table exposing protected builtins and a space for user symbols.

    Parameters
    ----------
    temperature_K : float
        Initial temperature for protected symbol `T`. Default 298.15.

    Behavior
    --------
    - Protected names (constants and T) cannot be shadowed by user code.
    - Built-in function names cannot be used as symbols.
    - The table stores only scalars for values (evaluator enforces this).
    - `update_temperature` is the sanctioned way for the application to change T.
    """

    temperature_K: float = 298.15
    _values: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _user: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize protected constants and T
        self._values = {
            "k_B": float(k_B),
            "h": float(h),
            "ħ": float(ħ),
            "N_A": float(N_A),
            "R": float(R),
            "Rkcal": float(Rkcal),
            "T": float(self.temperature_K),
        }
        # User namespace starts empty
        self._user = {}

    # --- lookup --------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Return True if a symbol (protected or user-defined) exists."""
        return name in self._values or name in self._user

    def get(self, name: str) -> float:
        """Get a symbol value, raising KeyError if unknown."""
        if name in self._values:
            return self._values[name]
        return self._user[name]  # raises KeyError as desired

    # --- mutation ------------------------------------------------------------

    def define_user(self, name: str, value: float, *, species_names: Optional[Set[str]] = None) -> None:
        """
        Define or overwrite a user symbol.

        Guards
        ------
        - Cannot use protected constant names or 'T'.
        - Cannot use built-in function names.
        - Cannot shadow a declared species.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("symbol name must be a non-empty string")
        if name in PROTECTED_NAMES:
            raise ValueError(f"attempted shadowing of protected symbol {name!r}")
        if name in PROTECTED_FUNCTION_NAMES:
            raise ValueError(f"attempted shadowing of builtin function {name!r}")
        if species_names and name in species_names:
            raise ValueError(f"attempted shadowing of species {name!r}")
        self._user[name] = float(value)

    def update_temperature(self, T: float) -> None:
        """
        System-only: update protected temperature `T`.

        This is not exposed to user algebra; evaluators should not allow 'let T = ...'.
        """
        self._values["T"] = float(T)

    # --- function exposure ---------------------------------------------------

    def functions(self) -> Mapping[str, Callable[..., float]]:
        """Return read-only view of built-in functions."""
        return BUILTIN_FUNCTIONS

    # --- discovery -----------------------------------------------------------

    def protected_names(self) -> Set[str]:
        """Return the set of protected symbol names."""
        return set(PROTECTED_NAMES)

    def user_names(self) -> Set[str]:
        """Return names of currently defined user symbols."""
        return set(self._user.keys())
