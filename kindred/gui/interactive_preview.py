"""Small immutable values exchanged by Slider gesture and plot lifecycles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from kindred.core.simulation_preparation import SimulationExecutionRequest


class SliderInteractionKind(str, Enum):
    INTERACTIVE_DRAG_SAMPLE = "interactive_drag_sample"
    EXACT_DRAG_RELEASE = "exact_drag_release"
    EXACT_DIRECT_CHANGE = "exact_direct_change"
    EXACT_TYPED_COMMIT = "exact_typed_commit"


@dataclass(frozen=True, slots=True)
class InteractivePreviewSampling:
    output_points: int = 48

    def __post_init__(self) -> None:
        if type(self.output_points) is not int:
            raise TypeError("output_points must be an int.")
        if not 10 <= self.output_points <= 250:
            raise ValueError("output_points must be between 10 and 250.")

    def lightweight_integration_points(self, configured_points: object) -> int:
        if type(configured_points) is not int:
            raise TypeError("configured integration points must be an int.")
        if configured_points < 10:
            raise ValueError("configured integration points must be >= 10.")
        return max(10, min(self.output_points, configured_points // 2))

    def validate_execution_grid(self, grid: object) -> None:
        if not isinstance(grid, Mapping):
            raise TypeError("Interactive Preview solver grid must be a mapping.")
        points = grid.get("N")
        if type(points) is not int:
            raise TypeError("Interactive Preview solver grid N must be an int.")
        if not 10 <= points <= self.output_points:
            raise ValueError(
                "Interactive Preview solver grid N must be between 10 and "
                f"{self.output_points}."
            )


def _required_text(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be nonblank.")
    return result


@dataclass(frozen=True, slots=True)
class InteractivePreviewTarget:
    set_id: str
    set_name: str
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", _required_text(self.set_id, "set_id"))
        object.__setattr__(self, "set_name", _required_text(self.set_name, "set_name"))


@dataclass(frozen=True, slots=True)
class InteractivePreviewRequest:
    tab_id: str
    targets: tuple[InteractivePreviewTarget, ...]
    sampling: InteractivePreviewSampling

    def __post_init__(self) -> None:
        tab = _required_text(self.tab_id, "tab_id")
        targets = tuple(self.targets)
        if not targets or any(not isinstance(item, InteractivePreviewTarget) for item in targets):
            raise TypeError("targets must contain at least one InteractivePreviewTarget.")
        ids = tuple(item.set_id for item in targets)
        if len(ids) != len(set(ids)):
            raise ValueError("Interactive Preview target set IDs must be unique.")
        if not isinstance(self.sampling, InteractivePreviewSampling):
            raise TypeError("sampling must be InteractivePreviewSampling.")
        for target in targets:
            self.sampling.validate_execution_grid(target.execution_request.solver_config.grid)
        object.__setattr__(self, "tab_id", tab)
        object.__setattr__(self, "targets", targets)

    @property
    def target_set_ids(self) -> tuple[str, ...]:
        return tuple(target.set_id for target in self.targets)
