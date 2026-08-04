# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)

__all__ = [
    "try_import_pyqtgraph",
    "is_pyqtgraph_available",
    "require_pyqtgraph",
    "pyqtgraph_import_error",
    "AxisSelectionCandidate",
    "PlotPresentationIntent",
]

_PYQTGRAPH_IMPORT_CACHE: Optional[Tuple[bool, Any, Optional[BaseException]]] = None


@dataclass(frozen=True, slots=True)
class AxisSelectionCandidate:
    """One typed axis identity paired with display-only text."""

    selection: AxisSeriesSelection
    label: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.selection,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
        ):
            raise TypeError(
                "Axis candidate selection must be a typed simulation or "
                "experimental selection."
            )
        label = str(self.label or "").strip()
        if not label:
            raise ValueError("Axis candidate label must be nonblank.")
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class PlotPresentationIntent:
    option: str
    enabled: bool

    def __post_init__(self) -> None:
        normalized = str(self.option or "").strip().lower()
        if normalized not in {
            "log_x",
            "log_y",
            "legend_visible",
            "x_inverted",
            "y_inverted",
        }:
            raise ValueError(f"Unknown persisted Plot option {self.option!r}.")
        if type(self.enabled) is not bool:
            raise TypeError("Plot presentation intent enabled must be a bool.")
        object.__setattr__(self, "option", normalized)


def try_import_pyqtgraph() -> Tuple[bool, Any, Optional[BaseException]]:
    """
    Attempt to import PyQtGraph and cache the result.

    This function is intentionally silent: it must not log at import time, so
    packaging tools (PyInstaller/Nuitka) can safely import Kindred modules during
    dependency analysis without producing misleading stderr output.
    """
    global _PYQTGRAPH_IMPORT_CACHE
    if _PYQTGRAPH_IMPORT_CACHE is not None:
        return _PYQTGRAPH_IMPORT_CACHE

    try:
        import pyqtgraph as pg  # type: ignore[import-not-found]
    except BaseException as exc:  # pragma: no cover - exercised via tests using import blocking
        _PYQTGRAPH_IMPORT_CACHE = (False, None, exc)
    else:
        _PYQTGRAPH_IMPORT_CACHE = (True, pg, None)
    return _PYQTGRAPH_IMPORT_CACHE


def pyqtgraph_import_error() -> Optional[BaseException]:
    """Return the cached PyQtGraph import exception, if any (without logging)."""
    ok, _pg, exc = try_import_pyqtgraph()
    if ok:
        return None
    return exc


def is_pyqtgraph_available() -> bool:
    ok, _pg, _exc = try_import_pyqtgraph()
    return bool(ok)


def require_pyqtgraph(*, context: str = "Kindred") -> Any:
    ok, pg, exc = try_import_pyqtgraph()
    if ok:
        return pg

    detail = ""
    if exc is not None:
        detail = f" (import failed: {exc!r})"
    raise ImportError(
        f"{context} requires PyQtGraph for plotting.{detail} "
        "Install with: pip install pyqtgraph"
    ) from exc
