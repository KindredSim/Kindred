"""
Canonical entry point for the Kindred desktop application.

This module launches the Kindred desktop GUI:

    python -m kindred
"""

from __future__ import annotations

import logging
import sys

from kindred.gui import startup as gui_startup
from kindred.io.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Launch the Kindred GUI and return the Qt event-loop exit code."""
    if len(sys.argv) > 1:
        print("Kindred launches the desktop GUI and does not accept command-line arguments.", file=sys.stderr)
        return 64

    setup_logging(level=None)
    startup_debug = gui_startup.startup_debug_enabled()
    cleanup_startup_diag = gui_startup.enable_startup_diagnostics(enabled=startup_debug)

    try:
        gui_startup.apply_pre_qapplication_startup(startup_debug=startup_debug)
        qt_core, qt_widgets, q_icon = gui_startup.ensure_qt_modules()

        if startup_debug:
            logger.warning("Startup phase: begin")

        gui_startup.log_plot_backend_startup(startup_debug=startup_debug)

        from kindred.gui.main_window import MainWindow

        qt_widgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            qt_core.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        if startup_debug:
            logger.warning("Startup phase: high-DPI policy set")

        app = qt_widgets.QApplication(sys.argv)
        qt_core.QCoreApplication.setOrganizationName("Kindred")
        qt_core.QCoreApplication.setApplicationName("Kindred")
        if startup_debug:
            logger.warning("Startup phase: QApplication created")

        gui_startup.apply_post_qapplication_startup(startup_debug=startup_debug)

        app.setStyle("Fusion")
        if startup_debug:
            logger.warning("Startup phase: app style set")

        icon = gui_startup.load_app_icon(q_icon)
        if icon is not None and not icon.isNull():
            app.setWindowIcon(icon)
        if startup_debug:
            logger.warning("Startup phase: icon loaded")

        win = gui_startup.construct_main_window(
            window_factory=MainWindow,
            startup_debug=startup_debug,
        )
        if startup_debug:
            logger.warning("Startup phase: MainWindow constructed")

        win.show()
        if startup_debug:
            logger.warning("Startup phase: MainWindow shown; entering event loop")

        return app.exec()
    finally:
        cleanup_startup_diag()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
