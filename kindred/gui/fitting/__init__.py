"""
Fitting UI components.

This package keeps fitting UI responsibilities split across focused owner
modules instead of a single catch-all orchestration surface.
"""

from __future__ import annotations

__all__ = ["FittingWindow", "GlobalFitLaunchContext", "launch_global_fit_session"]


def __getattr__(name: str):
    if name == "FittingWindow":
        from . import window as _window

        return getattr(_window, name)
    if name in __all__:
        from . import launch as _launch

        return getattr(_launch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
