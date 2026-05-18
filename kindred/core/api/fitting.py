"""
Authoritative fitting API.

- Import `fit_global` from this module when calling Kindred's multi-dataset fitting API.
- `GlobalFitResult` and `DatasetFitInfo` are the supported result types for that API.
- This module re-exports the current fitting implementation from
  `kindred.core.analysis.global_fitting`.
- `runtime_session`, `max_runtime_lanes`, and `runtime_ledger` are optional
  internal runtime-controller integration hooks used by the GUI fitting workflow.
- `contained_runtime=True` is an explicit public/core opt-in for a temporary
  contained runtime session. The default public path stays in-process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
    from kindred.core.fitting_completion import (
        FitDetailSection,
        FitDiagnostic,
        FitDiagnosticRemediation,
        GlobalFitCompletion,
    )

__all__ = [
    "fit_global",
    "GlobalFitResult",
    "DatasetFitInfo",
    "GlobalFitCompletion",
    "FitDiagnostic",
    "FitDiagnosticRemediation",
    "FitDetailSection",
]


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
    seed: Optional[int] = None,
    log10_params: Optional[Dict[str, bool]] = None,
    progress_callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    dataset_overrides: Optional[List[object]] = None,
    runtime_session: Optional[Any] = None,
    max_runtime_lanes: Optional[int] = None,
    runtime_ledger: Optional[Any] = None,
    contained_runtime: bool = False,
) -> Any:
    """Run global fitting through the authoritative core implementation.

    The runtime integration arguments are optional internal hooks for callers
    that already own a fitting runtime session. Ordinary callers should pass a
    fitting evaluator as the first positional argument.
    """
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
        seed=seed,
        log10_params=log10_params,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        dataset_overrides=dataset_overrides,
        runtime_session=runtime_session,
        max_runtime_lanes=max_runtime_lanes,
        runtime_ledger=runtime_ledger,
        contained_runtime=bool(contained_runtime),
    )


def __getattr__(name: str):
    if name in {"GlobalFitResult", "DatasetFitInfo"}:
        from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

        return {"GlobalFitResult": GlobalFitResult, "DatasetFitInfo": DatasetFitInfo}[name]
    if name in {"GlobalFitCompletion", "FitDiagnostic", "FitDiagnosticRemediation", "FitDetailSection"}:
        from kindred.core.fitting_completion import (
            FitDetailSection,
            FitDiagnostic,
            FitDiagnosticRemediation,
            GlobalFitCompletion,
        )

        return {
            "GlobalFitCompletion": GlobalFitCompletion,
            "FitDiagnostic": FitDiagnostic,
            "FitDiagnosticRemediation": FitDiagnosticRemediation,
            "FitDetailSection": FitDetailSection,
        }[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
