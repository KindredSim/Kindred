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

    # The limit is the admitted positive Fit evaluation budget.

    @property
    def remaining(self) -> int:
        return self.limit - self._nfev

    def claim(self, *, reserve: int = 0) -> int:
        """Claim one evaluation while leaving ``reserve`` claims untouched."""

        if self.remaining <= reserve:
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

    # The caller passes a positive budget (its limit above one, less one).
    dim = max(1, int(dim))

    popsize = int(math.floor(math.sqrt(budget / max(1, 2 * dim))))
    popsize = max(5, min(15, popsize))
    evals_per_gen = popsize * dim
    maxiter = max(0, budget // evals_per_gen - 1)

    return popsize, maxiter
