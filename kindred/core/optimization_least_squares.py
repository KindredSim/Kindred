from __future__ import annotations

import math
from typing import Any

__all__ = ["build_least_squares_kwargs"]


def build_least_squares_kwargs(
    *,
    ftol: float,
    xtol: float,
    max_nfev: int,
    verbose: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """Build SciPy ``least_squares`` kwargs from exact positive tolerances."""
    ftol_f = float(ftol)
    xtol_f = float(xtol)
    if not math.isfinite(ftol_f) or ftol_f <= 0.0:
        raise ValueError("ftol must be finite and positive")
    if not math.isfinite(xtol_f) or xtol_f <= 0.0:
        raise ValueError("xtol must be finite and positive")
    if "gtol" in extra:
        raise TypeError("gtol is not an exposed Fit tolerance.")

    kwargs: dict[str, Any] = {
        "ftol": ftol_f,
        "xtol": xtol_f,
        "max_nfev": int(max_nfev),
        "verbose": int(verbose),
    }
    kwargs.update(extra)
    return kwargs
