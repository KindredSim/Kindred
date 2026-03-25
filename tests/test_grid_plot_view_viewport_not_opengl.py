from __future__ import annotations

import pytest
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from kindred.gui.plot_config import require_pyqtgraph
from kindred.gui.widgets.grid_plot_view import GridPlotView

pytestmark = pytest.mark.gui


def test_grid_plot_view_forces_non_opengl_viewport(qtbot):
    pg = require_pyqtgraph(context="tests")
    # Simulate a user/system configuration that enables OpenGL globally.
    prev = bool(pg.getConfigOption("useOpenGL"))
    pg.setConfigOptions(useOpenGL=True)
    try:
        view = GridPlotView()
        qtbot.addWidget(view)

        gl = getattr(view, "_graphics_layout", None)
        assert gl is not None
        assert not isinstance(gl.viewport(), QOpenGLWidget)
    finally:
        pg.setConfigOptions(useOpenGL=prev)
