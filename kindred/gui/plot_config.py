# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)

__all__ = [
    "require_pyqtgraph",
    "AxisSelectionCandidate",
    "PlotPresentationIntent",
]


@dataclass(frozen=True, slots=True)
class AxisSelectionCandidate:
    """One typed axis identity paired with display-only text."""

    selection: AxisSeriesSelection
    label: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.selection,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
        ):
            raise TypeError(
                "Axis candidate selection must be a typed simulation or "
                "experimental selection."
            )
        label = str(self.label or "").strip()
        if not label:
            raise ValueError("Axis candidate label must be nonblank.")
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class PlotPresentationIntent:
    option: str
    enabled: bool

    def __post_init__(self) -> None:
        normalized = str(self.option or "").strip().lower()
        if normalized not in {
            "log_x",
            "log_y",
            "legend_visible",
            "x_inverted",
            "y_inverted",
        }:
            raise ValueError(f"Unknown persisted Plot option {self.option!r}.")
        if type(self.enabled) is not bool:
            raise TypeError("Plot presentation intent enabled must be a bool.")
        object.__setattr__(self, "option", normalized)


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
