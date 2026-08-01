"""
Canonical entry point for the Kindred desktop application.

This module launches the Kindred desktop GUI:

    python -m kindred
"""

from __future__ import annotations

import logging
import sys
from multiprocessing import freeze_support

from kindred.gui import startup as gui_startup
from kindred.io.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Launch the Kindred GUI and return the Qt event-loop exit code."""
    freeze_support()

    if len(sys.argv) > 1:
        print("Kindred launches the desktop GUI and does not accept command-line arguments.", file=sys.stderr)
        return 64

    setup_logging(level=None)
    startup_debug = gui_startup.startup_debug_enabled()
    cleanup_startup_diag = gui_startup.enable_startup_diagnostics(enabled=startup_debug)
    try:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ModuleNotFoundError:
            print(
                "PySide6 is required to launch the Kindred GUI. "
                "Install or reinstall Kindred with its standard GUI dependencies.",
                file=sys.stderr,
            )
            return 1

        gui_startup.apply_pre_qapplication_startup(
            qt_core=QtCore,
            startup_debug=startup_debug,
        )
        if startup_debug:
            logger.warning("Startup phase: begin")

        gui_startup.log_plot_backend_startup(startup_debug=startup_debug)

        from kindred.gui.main_window import MainWindow

        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        if startup_debug:
            logger.warning("Startup phase: high-DPI policy set")

        app = QtWidgets.QApplication(sys.argv)
        QtCore.QCoreApplication.setOrganizationName("Kindred")
        QtCore.QCoreApplication.setApplicationName("Kindred")
        if startup_debug:
            logger.warning("Startup phase: QApplication created")

        gui_startup.apply_post_qapplication_startup(startup_debug=startup_debug)
        app.setStyle("Fusion")
        if startup_debug:
            logger.warning("Startup phase: app style set")

        icon = gui_startup.load_app_icon(QtGui.QIcon)
        if icon is not None and not icon.isNull():
            app.setWindowIcon(icon)
        if startup_debug:
            logger.warning("Startup phase: icon loaded")

        win = MainWindow()
        if startup_debug:
            logger.warning("Startup phase: MainWindow constructed")

        win.show()
        if startup_debug:
            logger.warning("Startup phase: MainWindow shown; entering event loop")
        return app.exec()
    finally:
        cleanup_startup_diag()


if __name__ == "__main__":
    raise SystemExit(main())
