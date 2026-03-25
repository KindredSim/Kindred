"""
Authoritative fitting API.

Compatibility policy
--------------------
- Import `fit_global` from this module when calling Kindred's multi-dataset fitting API.
- `GlobalFitResult` and `DatasetFitInfo` are the supported result types for that API.
- This module is a stable facade over deeper analysis implementation modules; callers
  should not need to import `kindred.core.analysis.global_fitting` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

__all__ = ["fit_global", "GlobalFitResult", "DatasetFitInfo"]


def fit_global(*args: Any, **kwargs: Any) -> Any:
    from kindred.core.analysis.global_fitting import fit_global as _impl

    return _impl(*args, **kwargs)


def __getattr__(name: str):
    if name in {"GlobalFitResult", "DatasetFitInfo"}:
        from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

        return {"GlobalFitResult": GlobalFitResult, "DatasetFitInfo": DatasetFitInfo}[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
