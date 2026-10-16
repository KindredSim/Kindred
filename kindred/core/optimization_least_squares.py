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
    """Build SciPy ``least_squares`` kwargs from the admitted Fit tolerances."""
    kwargs: dict[str, Any] = {
        "ftol": float(ftol),
        "xtol": float(xtol),
        # SciPy tests optimality < gtol. The smallest positive float admits
        # only computed zero, without stopping on small concentration scales.
        "gtol": math.ulp(0.0),
        "max_nfev": int(max_nfev),
        "verbose": int(verbose),
    }
    kwargs.update(extra)
    return kwargs
