"""
Data analysis tools for chemical kinetics.

Modules:
- global_fitting: Global fitting across multiple datasets
"""

from __future__ import annotations

__all__ = [
    "fit_global",
    "GlobalFitResult",
    "DatasetFitInfo",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name in ["fit_global", "GlobalFitResult", "DatasetFitInfo"]:
        from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult, fit_global  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
