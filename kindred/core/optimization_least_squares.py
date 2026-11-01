from __future__ import annotations

import math
from typing import Any

__all__ = ["build_least_squares_kwargs"]


def build_least_squares_kwargs(
    *,
    ftol: float,
    xtol: float,
    max_nfev: int,
    **extra: Any,
) -> dict[str, Any]:
    """Build SciPy ``least_squares`` kwargs from the admitted Fit tolerances."""
    kwargs: dict[str, Any] = {
        "ftol": ftol,
        "xtol": xtol,
        # SciPy tests optimality < gtol. The smallest positive float admits
        # only computed zero, without stopping on small concentration scales.
        "gtol": math.ulp(0.0),
        "max_nfev": max_nfev,
        "verbose": 0,
    }
    kwargs.update(extra)
    return kwargs
