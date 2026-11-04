"""
Canonical entry point for the Kindred desktop application.

This module launches the Kindred desktop GUI:

    python -m kindred
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from multiprocessing import freeze_support

from kindred.io.logging import setup_logging

def _load_run_libraries() -> None:
    """Load what the first Run needs while the user works in the window."""
    from kindred.core.scipy_integrate import load_scipy_integrate
    from kindred.core.symbolic.backend import require_sympy

    require_sympy()
    load_scipy_integrate()


def main() -> int:
    """Launch the Kindred GUI and return the Qt event-loop exit code."""
    freeze_support()

    from kindred.gui import startup as gui_startup

    # Before numpy loads; spawned children inherit it.
    os.environ.update(gui_startup.SCIENTIFIC_THREAD_LIMITS)

    if len(sys.argv) > 1:
        print("Kindred launches the desktop GUI and does not accept command-line arguments.", file=sys.stderr)
        return 64

    setup_logging()
    # These lines mark where the session begins and whether it ended cleanly.
    logger = logging.getLogger("kindred")
    logger.info("Kindred session started")
    from PySide6 import QtCore, QtGui, QtWidgets

    gui_startup.apply_qt_startup_workarounds(qt_core=QtCore)

    from kindred.gui.main_window import MainWindow

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Fusion is the application's style override, so the theme's proxy style
    # wraps Fusion rather than the platform default.
    app = QtWidgets.QApplication([*sys.argv, "-style", "Fusion"])
    # One numeric-text convention: Python/repr decimal point, never the OS locale.
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    QtCore.QCoreApplication.setOrganizationName("Kindred")
    QtCore.QCoreApplication.setApplicationName("Kindred")

    icon = gui_startup.load_app_icon(QtGui.QIcon)
    if not icon.isNull():
        app.setWindowIcon(icon)

    win = MainWindow()
    win.show()
    # The first Run would otherwise wait for SciPy and SymPy to load; a
    # daemon thread so closing Kindred never waits for it.
    threading.Thread(target=_load_run_libraries, name="Kindred library preload", daemon=True).start()
    exit_code = app.exec()
    logger.info("Kindred session ended (exit code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
