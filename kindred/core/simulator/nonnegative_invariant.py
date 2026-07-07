from __future__ import annotations

import numpy as np

from kindred.core.exceptions import SolverError


def project_nonnegative_trajectory(
    Y: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, dict[str, object]]:
    Y2 = np.asarray(Y, float).copy()
    if Y2.size == 0:
        return Y2, {"projected": False, "count": 0, "max_negative_magnitude": 0.0}
    if not np.all(np.isfinite(Y2)):
        raise SolverError(f"{context} trajectory contains non-finite concentration values.")
    negative_mask = Y2 < 0.0
    count = int(np.count_nonzero(negative_mask))
    max_negative_magnitude = 0.0
    if count:
        max_negative_magnitude = float(np.max(np.abs(Y2[negative_mask])))
        Y2[negative_mask] = 0.0
    return Y2, {
        "projected": bool(count),
        "count": count,
        "max_negative_magnitude": max_negative_magnitude,
    }


def apply_nonnegative_trajectory_invariant(
    Y: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    projected, _provenance = project_nonnegative_trajectory(Y, context=context)
    return projected
