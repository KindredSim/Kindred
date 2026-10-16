# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

from dataclasses import dataclass
from kindred.core.project_presentation_state import AxisSeriesSelection

__all__ = [
    "AxisSelectionCandidate",
    "TIME_DISPLAY_UNITS",
]


TIME_DISPLAY_UNITS = ("fs", "ps", "ns", "µs", "ms", "s", "ks", "min", "h", "d", "yr")


@dataclass(frozen=True, slots=True)
class AxisSelectionCandidate:
    """One typed axis identity paired with display-only text."""

    selection: AxisSeriesSelection
    label: str

