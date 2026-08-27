"""Direct application and stable outer-shell theme forwarding."""

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
        theme_target: object,
        *,
        report_failure: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(application, QtWidgets.QApplication):
            raise TypeError("application must be a QApplication.")
        if not callable(getattr(theme_target, "set_theme", None)):
            raise TypeError(
                "theme_target must provide stable set_theme(dark_mode=...)."
            )
        self._theme_target = theme_target
        self._report_failure = report_failure
        self._applied_dark_mode: bool | None = None

    @property
    def applied_dark_mode(self) -> bool:
        """Return the last completely applied physical theme."""

        return bool(self._applied_dark_mode)

    def apply(
        self,
        dark_mode: bool,
        persist: Callable[[bool], None] | None = None,
    ) -> bool:
        """
        Apply one theme and optionally persist it.

        Physical application happens first.  The authoritative value advances
        only after physical application and persistence both succeed.  A
        failure restores the previous physical value.  Persistence owns
        restoring its own failed QSettings write; it is never called a second
        time as part of theme recovery.
        """
        requested = bool(dark_mode)
        recorded_previous = self._applied_dark_mode
        if recorded_previous is requested:
            return True
        previous = False if recorded_previous is None else recorded_previous

        physical_applied = False
        try:
            self._apply_physical(requested)
            physical_applied = True
            if persist is not None:
                persist(requested)
            self._applied_dark_mode = requested
        except Exception as exc:
            restored = self._restore_previous_physical(previous)
            if restored:
                self._applied_dark_mode = previous
            elif physical_applied:
                self._applied_dark_mode = requested
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

        logger.info("Applied %s theme", "dark" if requested else "light")
        return True

    def _apply_physical(self, dark_mode: bool) -> None:
        qdarktheme.setup_theme("dark" if dark_mode else "light")
        self._theme_target.set_theme(dark_mode=dark_mode)

    def _restore_previous_physical(
        self,
        previous: bool,
    ) -> bool:
        try:
            self._apply_physical(previous)
        except Exception:
            logger.exception("Failed to restore the previous physical theme.")
            return False
        return True
