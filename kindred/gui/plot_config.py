# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

import logging
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
]

_PYQTGRAPH_IMPORT_CACHE: Optional[Tuple[bool, Any, Optional[BaseException]]] = None


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

    _instance = None
    _backend = "pyqtgraph"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

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


def fix_pyqtgraph_stylehints_warning():
    """
    Fix PyQtGraph's QStyleHints UniqueConnection warning.

    PyQtGraph 0.13.7 connects to QStyleHints.colorSchemeChanged with
    Qt.UniqueConnection, which causes a warning in PySide6 because
    UniqueConnection cannot be enforced for Python slots.

    Since we cannot patch PySide6's signal connect method (it's read-only),
    this function modifies PyQtGraph's Qt module source to remove the
    UniqueConnection flag before it tries to connect.

    Must be called AFTER QApplication is created but BEFORE any PyQtGraph
    widgets are instantiated.
    """
    try:
        ok, _pg, _exc = try_import_pyqtgraph()
        if not ok:
            return

        import pyqtgraph.Qt as pgQt  # type: ignore[import-not-found]
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        # Check if colorSchemeChanged exists (Qt 6.5+)
        if not hasattr(app.styleHints(), 'colorSchemeChanged'):
            return

        # We need to prevent PyQtGraph from using UniqueConnection
        # Since the signal's connect is read-only, we'll pre-connect
        # the handler without UniqueConnection, so when PyQtGraph tries
        # to connect with UniqueConnection, it will fail silently (suppressed by contextlib.suppress)

        style_hints = app.styleHints()

        # Check if we've already fixed this
        if not hasattr(pgQt, '_kindred_colorscheme_fixed'):
            try:
                # Pre-connect the PyQtGraph handler without UniqueConnection
                if hasattr(pgQt, '_onColorSchemeChange'):
                    style_hints.colorSchemeChanged.connect(pgQt._onColorSchemeChange)
                    pgQt._kindred_colorscheme_fixed = True
                    logger.info("✓ Pre-connected PyQtGraph colorScheme handler to prevent UniqueConnection warning")
            except RuntimeError:
                # Already connected, that's fine
                pgQt._kindred_colorscheme_fixed = True
                logger.debug("ColorScheme handler already connected")

    except (ImportError, AttributeError) as e:
        # Qt 6.5+ feature, gracefully ignore if not available
        logger.debug(f"Could not fix PyQtGraph colorScheme connection: {e}")


def fix_pyqtgraph_csv_exporter_encoding():
    """
    Patch PyQtGraph's CSVExporter to write files as UTF-8 with BOM.

    PyQtGraph 0.13.7 opens CSV files with ``open(path, 'w', newline='')``,
    which uses the platform default encoding.  On Windows this is typically
    cp1252, causing ``UnicodeEncodeError`` when plot curve names contain
    non-ASCII characters (e.g. Greek letters in species names).

    The patch replaces the ``export`` method so that the file is opened with
    ``encoding='utf-8-sig'`` (UTF-8 with BOM, recognised by Excel on Windows).
    """
    ok, _pg, _exc = try_import_pyqtgraph()
    if not ok:
        return

    try:
        import csv
        import itertools

        import numpy as np
        from pyqtgraph.exporters.CSVExporter import CSVExporter  # type: ignore[import-not-found]

        if getattr(CSVExporter, "_kindred_encoding_patched", False):
            return

        _original_export = CSVExporter.export

        def _export_utf8(self, fileName=None):
            from pyqtgraph import PlotItem  # type: ignore[import-not-found]

            if not isinstance(self.item, PlotItem):
                raise TypeError("Must have a PlotItem selected for CSV export.")
            if fileName is None:
                self.fileSaveDialog(filter=["*.csv", "*.tsv"])
                return

            from pyqtgraph import ErrorBarItem  # type: ignore[import-not-found]

            for item in self.item.items:
                if isinstance(item, ErrorBarItem):
                    self._exportErrorBarItem(item)
                elif hasattr(item, "implements") and item.implements("plotData"):
                    self._exportPlotDataItem(item)

            sep = "," if self.params["separator"] == "comma" else "\t"
            columns = [col for dataset in self.data for col in dataset]
            with open(fileName, "w", encoding="utf-8-sig", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(self.header)
                for row in itertools.zip_longest(*columns, fillvalue=""):
                    writer.writerow([
                        item if isinstance(item, str)
                        else np.format_float_positional(item, precision=self.params["precision"])
                        for item in row
                    ])
            self.header.clear()
            self.data.clear()

        CSVExporter.export = _export_utf8
        CSVExporter._kindred_encoding_patched = True
        logger.debug("Patched PyQtGraph CSVExporter for UTF-8 encoding")

    except Exception as exc:
        logger.debug("Could not patch PyQtGraph CSVExporter: %s", exc)


def log_backend_info():
    """Log information about available backends."""
    ok, _pg, exc = try_import_pyqtgraph()
    logger.info("=" * 60)
    logger.info("Plot Backend Configuration")
    logger.info("=" * 60)
    logger.info(f"Available backends: {', '.join(available_backends())}")
    logger.info(f"PyQtGraph available: {ok}")
    logger.info(f"Selected backend: {PlotBackend.get_backend_name()}")
    logger.info("=" * 60)

    if ok:
        logger.info("Using PyQtGraph for plotting.")
        return

    logger.error("PyQtGraph import failed; plotting is unavailable and GUI startup will fail.")
    if exc is not None:
        logger.debug("PyQtGraph import exception:", exc_info=exc)
