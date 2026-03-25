from __future__ import annotations

from typing import Any, Callable

__all__ = ["load_scipy_integrate"]


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

