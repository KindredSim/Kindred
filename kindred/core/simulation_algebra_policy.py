from __future__ import annotations

from enum import Enum


class SimulationAlgebraPolicy(str, Enum):
    """Caller-owned Algebra handling policy for simulation finalization."""

    GUI_BEST_EFFORT = "gui_best_effort"
    FITTING_STRICT = "fitting_strict"
