# kindred/gui/widgets/pyqtgraph_plot_panel.py
"""
Compatibility wrapper for the PyQtGraph plot panel.

The real implementation lives in `pyqtgraph_plot_panel_impl.py` and is loaded
only after `kindred.gui.plot_config` has confirmed PyQtGraph is importable.

This module must be safe to import during packaging dependency analysis
(PyInstaller/Nuitka) and therefore must not attempt to import PyQtGraph or emit
logging warnings/errors at import time.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ..ui_helpers import make_pyqtgraph_fallback_widget

logger = logging.getLogger(__name__)

__all__ = ["PyQtGraphPlotPanel"]


class PyQtGraphPlotPanel(QtWidgets.QWidget):
    """Stub class used when the real PyQtGraph panel is not loaded."""

    seriesVisibilityChanged = QtCore.Signal(str, bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(
            make_pyqtgraph_fallback_widget(
                self,
                text=(
                    "PyQtGraph plot panel is unavailable.\n\n"
                    "This usually means PyQtGraph failed to import in the current environment.\n"
                    "Install with: pip install pyqtgraph"
                ),
            )
        )

    def set_data(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def set_theme(self, *args, **kwargs):
        pass

    def visible_series(self):
        return []

    def set_series_visible(self, *args, **kwargs) -> None:
        pass

    def visible(self, *args, **kwargs) -> bool:
        return False

    def set_overlay_catalog(self, *args, **kwargs):
        pass

    def active_overlays(self):
        return []

    def overlay_snapshot(self):
        return []
