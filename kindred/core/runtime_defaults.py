from __future__ import annotations

USE_ANALYTIC_JACOBIAN_DEFAULT = True
# Hard ceiling for bounded Run worker fan-out.
MAX_PARALLEL_WORKERS_CEILING = 60


__all__ = [
    "USE_ANALYTIC_JACOBIAN_DEFAULT",
    "MAX_PARALLEL_WORKERS_CEILING",
]
