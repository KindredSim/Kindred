"""Budget and configuration helpers for SciPy Differential Evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple


class DifferentialEvolutionBudgetExhausted(RuntimeError):
    """Signal that no unreserved Differential Evolution evaluation remains."""


@dataclass(slots=True)
class DifferentialEvolutionEvaluationBudget:
    """Own the hard objective-evaluation claims for one DE execution."""

    limit: int
    _nfev: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.limit = int(self.limit)
        if self.limit < 1:
            raise ValueError("Differential Evolution max_nfev must be positive.")

    @property
    def remaining(self) -> int:
        return self.limit - self._nfev

    def claim(self, *, reserve: int = 0) -> int:
        """Claim one evaluation while leaving ``reserve`` claims untouched."""

        reserved = int(reserve)
        if reserved < 0:
            raise ValueError("Differential Evolution evaluation reserve cannot be negative.")
        if self.remaining <= reserved:
            raise DifferentialEvolutionBudgetExhausted(
                "Differential Evolution reached its maximum objective-evaluation budget."
            )
        self._nfev += 1
        return self._nfev


def compute_de_popsize_maxiter(*, budget: int, dim: int) -> Tuple[int, int]:
    """Return (popsize, maxiter) for SciPy differential_evolution.

    SciPy's DE uses a population size of ``popsize * dim`` and a maximum of
    ``(maxiter + 1) * popsize * dim`` objective evaluations. ``budget`` is the
    portion available to SciPy after the caller reserves its final-candidate
    evaluation. The execution-local evaluation claimant remains the exact hard
    authority, including when SciPy's minimum initial population exceeds this
    estimate.
    """

    dim = max(1, int(dim))
    budget = int(budget)
    if budget < 1:
        raise ValueError("Differential Evolution optimizer budget must be positive.")

    popsize = int(math.floor(math.sqrt(budget / max(1, 2 * dim))))
    popsize = max(5, min(15, popsize))
    evals_per_gen = popsize * dim
    maxiter = max(0, budget // evals_per_gen - 1)

    return popsize, maxiter
