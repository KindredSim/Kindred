from __future__ import annotations

import math
from typing import Any

__all__ = ["build_least_squares_kwargs"]


def build_least_squares_kwargs(
    *,
    ftol: float,
    xtol: float,
    max_nfev: int,
    gtol: float | None = None,
    verbose: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """
    Build SciPy `least_squares` kwargs with a hard optimization contract.

    Contract:
    - `gtol` must strictly mirror `ftol` (UI stability invariant).
    - When `gtol` is not provided, it is set to `ftol`.
    """
    ftol_f = float(ftol)
    xtol_f = float(xtol)
    if not math.isfinite(ftol_f) or not math.isfinite(xtol_f):
        raise ValueError("ftol/xtol must be finite")

    if gtol is None:
        gtol_f = ftol_f
    else:
        gtol_f = float(gtol)
        if not math.isfinite(gtol_f):
            raise ValueError("gtol must be finite")
        if gtol_f != ftol_f:
            raise ValueError("gtol must equal ftol (architectural invariant)")

    kwargs: dict[str, Any] = {
        "ftol": ftol_f,
        "xtol": xtol_f,
        "gtol": gtol_f,
        "max_nfev": int(max_nfev),
        "verbose": int(verbose),
    }
    kwargs.update(extra)
    return kwargs

