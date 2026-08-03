# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "PlotBackend",
    "get_plot_panel_class",
    "available_backends",
    "try_import_pyqtgraph",
    "is_pyqtgraph_available",
    "require_pyqtgraph",
    "pyqtgraph_import_error",
    "PlotPresentationIntent",
]

_PYQTGRAPH_IMPORT_CACHE: Optional[Tuple[bool, Any, Optional[BaseException]]] = None


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


def available_backends() -> list[str]:
    return ["pyqtgraph"] if is_pyqtgraph_available() else []


class PlotBackend:
    """
    Plot backend configuration.

    Manages PyQtGraph plotting backend (GPU-accelerated).

    Attributes
    ----------
    backend : str
        Current backend: always 'pyqtgraph'
    """

    _backend = "pyqtgraph"

    @classmethod
    def set_backend(cls, backend: str) -> None:
        """
        Set plot backend.

        Parameters
        ----------
        backend : str
            Must be 'pyqtgraph' (or 'auto' which resolves to pyqtgraph)

        Raises
        ------
        ValueError
            If backend is not 'pyqtgraph' or 'auto'
        """
        if backend == "auto" or backend == "pyqtgraph":
            if not is_pyqtgraph_available():
                raise ValueError(
                    "PyQtGraph backend not available. "
                    "Install with: pip install pyqtgraph"
                )
            cls._backend = "pyqtgraph"
            logger.info("Plot backend set to PyQtGraph (GPU-accelerated)")
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Only 'pyqtgraph' is supported."
            )

    @classmethod
    def get_backend(cls) -> str:
        """
        Get current backend.

        Returns
        -------
        str
            Always 'pyqtgraph'
        """
        return "pyqtgraph"

    @classmethod
    def get_backend_name(cls) -> str:
        """Get human-readable backend name."""
        return "PyQtGraph (GPU-accelerated)"


def get_plot_panel_class():
    """
    Get the configured plot panel class based on PyQtGraph availability.

    Returns
    -------
    class
        PyQtGraphPlotPanel

    Examples
    --------
    >>> PlotPanelClass = get_plot_panel_class()
    >>> plot = PlotPanelClass()
    """
    require_pyqtgraph(context="Kindred GUI")

    from kindred.gui.widgets.pyqtgraph_plot_panel_impl import PyQtGraphPlotPanel
    logger.debug("Using PyQtGraph plot panel")
    return PyQtGraphPlotPanel
