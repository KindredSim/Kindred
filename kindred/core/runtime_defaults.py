from __future__ import annotations

import os

USE_SPARSE_JACOBIAN_DEFAULT = True
WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT = True
LIMIT_BLAS_THREADS_PER_WORKER_DEFAULT = True
# Hard ceiling for Windows WaitForMultipleObjects handle limit.
MAX_PARALLEL_WORKERS_CEILING = 60
CONTAINED_CHILD_BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def contained_child_blas_thread_env(*, enabled: bool = True) -> dict[str, str]:
    if not bool(enabled):
        return {}
    return {name: "1" for name in CONTAINED_CHILD_BLAS_THREAD_ENV_VARS}


def _compute_max_parallel_batch_workers_default(cpu_count: int | None = None) -> int:
    detected_cpu_count = os.cpu_count() if cpu_count is None else cpu_count
    return min(max(1, (detected_cpu_count or 1) - 1), 16)


MAX_PARALLEL_BATCH_WORKERS_DEFAULT = _compute_max_parallel_batch_workers_default()
BATCH_RUNTIME_LANE_BUDGET_DEFAULT = MAX_PARALLEL_BATCH_WORKERS_DEFAULT
RESULT_CACHE_CAP_DEFAULT = 1000
PREVIEW_CACHE_CAP_DEFAULT = 1000

__all__ = [
    "USE_SPARSE_JACOBIAN_DEFAULT",
    "WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT",
    "LIMIT_BLAS_THREADS_PER_WORKER_DEFAULT",
    "CONTAINED_CHILD_BLAS_THREAD_ENV_VARS",
    "contained_child_blas_thread_env",
    "MAX_PARALLEL_WORKERS_CEILING",
    "MAX_PARALLEL_BATCH_WORKERS_DEFAULT",
    "BATCH_RUNTIME_LANE_BUDGET_DEFAULT",
    "RESULT_CACHE_CAP_DEFAULT",
    "PREVIEW_CACHE_CAP_DEFAULT",
]
