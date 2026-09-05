from __future__ import annotations

import math
import sys
from numbers import Real
from typing import Any, Callable

__all__ = [
    "SCIPY_MIN_RTOL",
    "load_scipy_integrate",
    "validate_scipy_rtol",
]

SCIPY_MIN_RTOL: float = 100.0 * sys.float_info.epsilon


def validate_scipy_rtol(value: object, *, field: str = "rtol") -> float:
    """Validate one relative tolerance accepted by the pinned SciPy backend."""

    if type(value) is bool or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive.")
    if parsed < SCIPY_MIN_RTOL:
        raise ValueError(
            f"{field} must be at least {SCIPY_MIN_RTOL}."
        )
    return parsed


def load_scipy_integrate() -> Callable[..., Any]:
    """
    Load SciPy integration entry points with a consistent error message.

    Kindred requires SciPy for ODE integration. This
    helper keeps SciPy imports off the module import path for GUI modules and
    headless tooling that only need integration when solving an ODE.
    """
    try:
        from scipy.integrate import solve_ivp  # type: ignore
    except ImportError as exc:  # pragma: no cover - SciPy is a pinned dependency
        raise ImportError(
            "scipy is required for ODE integration. Install with: pip install scipy"
        ) from exc
    return solve_ivp
