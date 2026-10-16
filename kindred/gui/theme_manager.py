"""Direct application and stable outer-shell theme forwarding."""

from __future__ import annotations

import logging
from collections.abc import Callable

import qdarktheme
from PySide6 import QtGui

logger = logging.getLogger(__name__)

__all__ = [
    "ThemeManager", "semantic_color", "plot_background_color",
    "plot_foreground_color", "readable_plot_color",
]


def semantic_color(role: str, dark_mode: bool) -> QtGui.QColor:
    """Return readable accent text for the current light or dark surface."""
    light, dark = {
        "warning": ("#795400", "#ffd478"),
        "error": ("#b42332", "#ff9aa5"),
        "success": ("#216739", "#8cdda4"),
        "muted": ("#555b63", "#bac1ca"),
    }[role]
    return QtGui.QColor(dark if dark_mode else light)


def plot_background_color(dark_mode: bool) -> QtGui.QColor:
    return QtGui.QColor("#1e1e1e" if dark_mode else "#ffffff")


def plot_foreground_color(dark_mode: bool) -> QtGui.QColor:
    return QtGui.QColor("#dddddd" if dark_mode else "#222222")


def readable_plot_color(color: QtGui.QColor, dark_mode: bool) -> QtGui.QColor:
    """Adjust generated colors for plot contrast while retaining their hue."""
    def luminance(value: QtGui.QColor) -> float:
        channels = [channel / 12.92 if channel <= 0.04045 else
                    ((channel + 0.055) / 1.055) ** 2.4
                    for channel in (value.redF(), value.greenF(), value.blueF())]
        return sum(channel * weight for channel, weight in
                   zip(channels, (0.2126, 0.7152, 0.0722)))

    background = luminance(plot_background_color(dark_mode))
    adjusted = QtGui.QColor(color)
    for _ in range(32):
        foreground = luminance(adjusted)
        contrast = (max(background, foreground) + 0.05) / (min(background, foreground) + 0.05)
        if contrast >= 4.5:
            break
        adjusted = adjusted.lighter(110) if dark_mode else adjusted.darker(110)
    return adjusted


class ThemeManager:
    """Own and apply the application's one authoritative theme."""

    def __init__(
        self,
        theme_target: object,
        *,
        report_failure: Callable[[str], None],
    ) -> None:
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
    ) -> None:
        """
        Apply one theme and optionally persist it.

        Physical application happens first.  The authoritative value advances
        only after physical application and persistence both succeed.  A
        persistence failure restores the previous physical value.  Persistence owns
        restoring its own failed QSettings write; it is never called a second
        time as part of theme recovery.
        """
        requested = bool(dark_mode)
        recorded_previous = self._applied_dark_mode
        if recorded_previous is requested:
            return
        previous = False if recorded_previous is None else recorded_previous

        self._apply_physical(requested)
        if persist is not None:
            try:
                persist(requested)
            except OSError as exc:
                self._apply_physical(previous)
                self._applied_dark_mode = previous
                message = (
                    f"Could not apply the {'dark' if requested else 'light'} theme: {exc}"
                )
                logger.warning(message, exc_info=True)
                self._report_failure(message)
                return
        self._applied_dark_mode = requested
        logger.info("Applied %s theme", "dark" if requested else "light")

    def _apply_physical(self, dark_mode: bool) -> None:
        qdarktheme.setup_theme(
            "dark" if dark_mode else "light",
            custom_colors={"[light]": {
                "background": "#eef0f3",
                "input.background": "#ffffff",
                "border": "#a8afb9",
            }},
        )
        self._theme_target.set_theme(dark_mode=dark_mode)
