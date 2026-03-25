"""Shared helpers for configuring SciPy Differential Evolution.

This module centralizes how we translate an evaluation budget into SciPy DE
parameters so local/global fitting paths cannot drift.
"""

from __future__ import annotations

import math
from typing import Tuple


def compute_de_popsize_maxiter(*, budget: int, dim: int) -> Tuple[int, int]:
    """Return (popsize, maxiter) for SciPy differential_evolution.

    SciPy's DE uses a population size of ``popsize * dim`` and a maximum of
    ``(maxiter + 1) * popsize * dim`` objective evaluations. The heuristic here
    mirrors the pre-existing local and global fitting logic.
    """

    dim = max(1, int(dim))
    budget = max(1, int(budget))

    popsize = int(math.floor(math.sqrt(budget / max(1, 2 * dim))))
    popsize = max(5, min(15, popsize))
    evals_per_gen = popsize * dim
    maxiter = max(1, budget // evals_per_gen - 1)

    return popsize, maxiter

