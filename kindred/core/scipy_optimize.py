from __future__ import annotations

from typing import Any, Callable

__all__ = ["load_scipy_optimize"]


def load_scipy_optimize() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Load SciPy optimization entry points with a consistent error message.

    Kindred requires SciPy for fitting. This helper
    keeps SciPy imports off the module import path for GUI modules that only need
    fitting when the user runs an optimization.
    """
    try:
        from scipy.optimize import differential_evolution, least_squares
    except ImportError as exc:  # pragma: no cover - SciPy is a pinned dependency
        raise ImportError("scipy is required for fitting. Install with: pip install scipy") from exc
    return least_squares, differential_evolution

