"""
Authoritative fitting API.

- Import `fit_global` from this module when calling Kindred's multi-dataset fitting API.
- `GlobalFitResult` and `DatasetFitInfo` are the supported result types for that API.
- This module re-exports the current fitting implementation from
  `kindred.core.analysis.global_fitting`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

__all__ = ["fit_global", "GlobalFitResult", "DatasetFitInfo"]


def fit_global(
    fit_evaluator: Any,
    datasets: List[object],
    shared_params: Dict[str, float],
    dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_variable_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = "trf",
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    parallel_enabled: bool = False,
    max_parallel_workers: int = 1,
    limit_blas_threads: bool = True,
    seed: Optional[int] = None,
    log10_params: Optional[Dict[str, bool]] = None,
    progress_callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    dataset_overrides: Optional[List[object]] = None,
) -> Any:
    from kindred.core.analysis.global_fitting import fit_global as _impl

    return _impl(
        fit_evaluator,
        datasets,
        shared_params,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
        bounds=bounds,
        weights=weights,
        method=method,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        parallel_enabled=parallel_enabled,
        max_parallel_workers=max_parallel_workers,
        limit_blas_threads=limit_blas_threads,
        seed=seed,
        log10_params=log10_params,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        dataset_overrides=dataset_overrides,
    )


def __getattr__(name: str):
    if name in {"GlobalFitResult", "DatasetFitInfo"}:
        from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

        return {"GlobalFitResult": GlobalFitResult, "DatasetFitInfo": DatasetFitInfo}[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
