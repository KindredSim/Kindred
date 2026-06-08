"""Data analysis tools for chemical kinetics."""

from __future__ import annotations

__all__ = [
    "fit_global",
    "GlobalFitResult",
    "DatasetFitInfo",
]


def __getattr__(name: str):
    if name in {"fit_global", "GlobalFitResult", "DatasetFitInfo"}:
        from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult, fit_global

        return {"fit_global": fit_global, "GlobalFitResult": GlobalFitResult, "DatasetFitInfo": DatasetFitInfo}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
