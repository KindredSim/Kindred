"""Small immutable values exchanged by Slider gesture and plot lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from kindred.core.simulation_preparation import SimulationExecutionRequest


class SliderInteractionKind(str, Enum):
    INTERACTIVE_DRAG_SAMPLE = "interactive_drag_sample"
    EXACT_DRAG_RELEASE = "exact_drag_release"
    EXACT_DIRECT_CHANGE = "exact_direct_change"
    EXACT_TYPED_COMMIT = "exact_typed_commit"


@dataclass(frozen=True, slots=True)
class InteractivePreviewTarget:
    set_id: str
    set_name: str
    execution_request: SimulationExecutionRequest


@dataclass(frozen=True, slots=True)
class InteractivePreviewRequest:
    tab_id: str
    targets: tuple[InteractivePreviewTarget, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not targets:
            raise TypeError("targets must contain at least one InteractivePreviewTarget.")
        ids = tuple(item.set_id for item in targets)
        if len(ids) != len(set(ids)):
            raise ValueError("Interactive Preview target set IDs must be unique.")
        object.__setattr__(self, "targets", targets)

    @property
    def target_set_ids(self) -> tuple[str, ...]:
        return tuple(target.set_id for target in self.targets)
