"""Centralized theme management for Qt widgets and plots."""

from __future__ import annotations

import logging
from typing import Optional

import qdarktheme

logger = logging.getLogger(__name__)


class ThemeManager:
    """
    Apply dark/light themes consistently across Qt widgets and plotting surfaces.

    Parameters
    ----------
    plot_tabs : PlotTabsWidget
        Container hosting the simulation plot, dataset tabs, and grid view.
    """

    def __init__(self, plot_tabs) -> None:
        self._plot_tabs = plot_tabs
        self._dark_mode: Optional[bool] = None

    def apply(self, dark_mode: bool) -> None:
        """Apply the requested theme (True = dark, False = light)."""
        is_dark = bool(dark_mode)
        if self._dark_mode == is_dark:
            logger.debug("Theme already set to %s; skipping", "dark" if is_dark else "light")
            return
        self._dark_mode = is_dark

        theme_str = "dark" if is_dark else "light"
        try:
            qdarktheme.setup_theme(theme_str)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to apply qdarktheme %s theme: %s", theme_str, exc)

        # Apply theme to PyQtGraph plots via refresh() which calls set_dark_mode() on widgets
        self.refresh(is_dark)
        logger.info("Applied %s theme", theme_str)

    def refresh(self, is_dark: Optional[bool] = None) -> None:
        """Refresh registered plot widgets to adopt the current theme."""
        if is_dark is not None:
            self._dark_mode = bool(is_dark)
        try:
            self._refresh_plot_widget(getattr(self._plot_tabs, "_main_plot", None))
            for _, panel in getattr(self._plot_tabs, "_dataset_plots", []):
                self._refresh_plot_widget(panel)

            grid_view = getattr(self._plot_tabs, "_grid_view", None)
            if grid_view is not None and hasattr(grid_view, "refresh"):
                grid_view.refresh()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to refresh some plots after theme change: %s", exc)

    def is_dark(self) -> bool:
        """Return True if dark mode is currently active."""
        return self._dark_mode is True

    def _refresh_plot_widget(self, plot: Optional[object]) -> None:
        if plot is None:
            return
        enabled = bool(self._dark_mode)
        # PyQtGraph widgets use set_dark_mode()
        if hasattr(plot, "set_dark_mode"):
            plot.set_dark_mode(enabled=enabled)
        elif hasattr(plot, "set_theme"):
            plot.set_theme(dark_mode=enabled)
