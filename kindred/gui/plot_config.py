# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    VesselVolumeAxisSelection,
    SimulationSeriesAxisSelection,
)

__all__ = [
    "AxisSelectionCandidate",
    "TIME_DISPLAY_UNITS",
    "require_pyqtgraph",
]


TIME_DISPLAY_UNITS = ("fs", "ps", "ns", "µs", "ms", "s", "ks", "min", "h", "d", "yr")


@dataclass(frozen=True, slots=True)
class AxisSelectionCandidate:
    """One typed axis identity paired with display-only text."""

    selection: AxisSeriesSelection
    label: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.selection,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection, VesselVolumeAxisSelection),
        ):
            raise TypeError(
                "Axis candidate selection must be a typed simulation or "
                "experimental selection."
            )
        label = str(self.label or "").strip()
        if not label:
            raise ValueError("Axis candidate label must be nonblank.")
        object.__setattr__(self, "label", label)


def require_pyqtgraph(*, context: str = "Kindred") -> Any:
    """Return the required plotting backend or raise one dependency error."""

    try:
        import pyqtgraph as pg  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(
            f"{context} requires PyQtGraph for plotting "
            f"(import failed: {exc!r}). Install with: pip install pyqtgraph"
        ) from exc
    return pg
