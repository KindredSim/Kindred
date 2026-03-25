"""
Shared GUI utilities for Kindred.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["BusyCursor"]


class BusyCursor:
    """
    Context manager for showing a busy cursor during blocking operations.

    Usage:
        with BusyCursor():
            do_work()
    """
    def __enter__(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        QtWidgets.QApplication.restoreOverrideCursor()
        return False
