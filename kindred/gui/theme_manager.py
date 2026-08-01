"""Direct application and shared-plot theme ownership."""

from __future__ import annotations

import logging
from collections.abc import Callable

import qdarktheme
from PySide6 import QtWidgets

logger = logging.getLogger(__name__)

__all__ = ["ThemeManager"]


class ThemeManager:
    """Own and apply the application's one authoritative theme."""

    def __init__(
        self,
        application: QtWidgets.QApplication,
        plot_container: object,
        *,
        report_failure: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(application, QtWidgets.QApplication):
            raise TypeError("application must be a QApplication.")
        if not callable(getattr(plot_container, "set_theme", None)):
            raise TypeError("plot_container must provide set_theme(dark_mode=...).")
        self._application = application
        self._plot_container = plot_container
        self._report_failure = report_failure
        self._applied_dark_mode: bool | None = None

    def apply(
        self,
        dark_mode: bool,
        persist: Callable[[bool], None] | None = None,
    ) -> bool:
        """
        Apply one theme and optionally persist it.

        Physical application happens first.  The authoritative value advances
        only after physical application and persistence both succeed.  A
        failure restores the previous physical and persisted value exactly
        once.
        """
        requested = bool(dark_mode)
        recorded_previous = self._applied_dark_mode
        if recorded_previous is requested:
            return True
        previous = False if recorded_previous is None else recorded_previous

        try:
            self._apply_physical(requested)
            if persist is not None:
                persist(requested)
        except Exception as exc:
            self._restore_previous_once(previous, persist)
            self._applied_dark_mode = previous
            message = (
                f"Could not apply the {'dark' if requested else 'light'} theme: {exc}"
            )
            logger.warning(message, exc_info=True)
            if self._report_failure is not None:
                try:
                    self._report_failure(message)
                except Exception:
                    logger.exception("Theme failure reporter raised an exception.")
            return False

        self._applied_dark_mode = requested
        logger.info("Applied %s theme", "dark" if requested else "light")
        return True

    def _apply_physical(self, dark_mode: bool) -> None:
        qdarktheme.setup_theme("dark" if dark_mode else "light")
        self._plot_container.set_theme(dark_mode=dark_mode)

    def _restore_previous_once(
        self,
        previous: bool,
        persist: Callable[[bool], None] | None,
    ) -> None:
        try:
            self._apply_physical(previous)
        except Exception:
            logger.exception("Failed to restore the previous physical theme.")
        if persist is not None:
            try:
                persist(previous)
            except Exception:
                logger.exception("Failed to restore the previous persisted theme.")
