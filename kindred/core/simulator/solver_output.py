from __future__ import annotations

import numpy as np

__all__ = ["solver_trajectory_array"]


def solver_trajectory_array(sol_y: object, *, y0: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    sample_count = int(np.asarray(t_out, dtype=float).reshape(-1).size)
    species_count = int(np.asarray(y0, dtype=float).reshape(-1).size)
    Y = np.asarray(sol_y, dtype=float)
    if Y.ndim == 2:
        return Y
    if sample_count == 0:
        return np.empty((species_count, 0), dtype=float)
    if Y.ndim == 1 and species_count == 1 and Y.size == sample_count:
        return Y.reshape(1, sample_count)
    if Y.ndim == 1 and sample_count == 1 and Y.size == species_count:
        return Y.reshape(species_count, 1)
    return Y.reshape(species_count, sample_count)
