from __future__ import annotations

import numpy as np

from kindred.core.exceptions import SolverError


def apply_nonnegative_trajectory_invariant(
    Y: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    Y2 = np.asarray(Y, float).copy()
    if Y2.size == 0:
        return Y2
    if not np.all(np.isfinite(Y2)):
        raise SolverError(f"{context} trajectory contains non-finite concentration values.")
    Y2[Y2 < 0.0] = 0.0
    return Y2
