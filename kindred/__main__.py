"""
Canonical entry point for the Kindred desktop application.

This module launches the Kindred desktop GUI:

    python -m kindred
"""

from __future__ import annotations

import sys
from multiprocessing import freeze_support

from kindred.gui import startup as gui_startup
from kindred.io.logging import setup_logging

def main() -> int:
    """Launch the Kindred GUI and return the Qt event-loop exit code."""
    freeze_support()

    if len(sys.argv) > 1:
        print("Kindred launches the desktop GUI and does not accept command-line arguments.", file=sys.stderr)
        return 64

    setup_logging(level=None)
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ModuleNotFoundError:
        print(
            "PySide6 is required to launch the Kindred GUI. "
            "Install or reinstall Kindred with its standard GUI dependencies.",
            file=sys.stderr,
        )
        return 1

    gui_startup.apply_pre_qapplication_startup(qt_core=QtCore)

    from kindred.gui.main_window import MainWindow

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.setOrganizationName("Kindred")
    QtCore.QCoreApplication.setApplicationName("Kindred")
    app.setStyle("Fusion")

    icon = gui_startup.load_app_icon(QtGui.QIcon)
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
