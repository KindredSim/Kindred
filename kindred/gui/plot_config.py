# kindred/gui/plot_config.py
"""Plot backend configuration and selection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)

__all__ = [
    "AxisSelectionCandidate",
    "CONCENTRATION_DISPLAY_UNIT_FACTORS",
    "TIME_DISPLAY_UNIT_FACTORS",
    "require_pyqtgraph",
]


TIME_DISPLAY_UNIT_FACTORS = MappingProxyType(
    {
        "fs": 1e-15,
        "ps": 1e-12,
        "ns": 1e-9,
        "µs": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
        "ks": 1e3,
        "min": 60.0,
        "h": 3600.0,
        "d": 86_400.0,
        "yr": 31_557_600.0,
    }
)

CONCENTRATION_DISPLAY_UNIT_FACTORS = MappingProxyType(
    {
        "fM": 1e-15,
        "pM": 1e-12,
        "nM": 1e-9,
        "µM": 1e-6,
        "mM": 1e-3,
        "M": 1.0,
    }
)


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
