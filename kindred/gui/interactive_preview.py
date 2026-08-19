"""Small immutable values exchanged by Slider gesture and plot lifecycles.

Interactive previews are transient.  Their only currentness value is the
integer operation generation owned by :class:`InteractivePreviewOwner`;
physical rendering never creates a second identity or acknowledgement route.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math

from kindred.core.simulation_preparation import SimulationExecutionRequest
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.solver_types import SimulationPresentationBoundary


class SliderInteractionKind(str, Enum):
    INTERACTIVE_DRAG_SAMPLE = "interactive_drag_sample"
    EXACT_DRAG_RELEASE = "exact_drag_release"
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


def _finite_tuple(values: Iterable[float], *, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain finite values and not be empty.")
    return result


@dataclass(frozen=True, slots=True)
class InteractivePreviewTarget:
    set_id: str
    set_name: str
    prepared_dsl: PreparedDslRequestSet
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", _required_text(self.set_id, "set_id"))
        object.__setattr__(self, "set_name", _required_text(self.set_name, "set_name"))
        if self.execution_request.prepared_dsl is not self.prepared_dsl:
            raise ValueError("Interactive Preview request must retain its accepted prepared owner.")


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


@dataclass(frozen=True, slots=True)
class InteractivePreviewSeries:
    series_id: str
    label: str
    values: tuple[float, ...]
    color: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", _required_text(self.series_id, "series_id"))
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(self, "values", _finite_tuple(self.values, field="series values"))
        if self.color is not None:
            object.__setattr__(self, "color", _required_text(self.color, "color"))


@dataclass(frozen=True, slots=True)
class InteractivePreviewFrame:
    tab_id: str
    set_id: str
    generation: int
    target_set_ids: tuple[str, ...]
    x_values: tuple[float, ...]
    segment_ids: tuple[int, ...]
    segment_keys: tuple[str, ...]
    discontinuity_ids: tuple[int, ...]
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...]
    series: tuple[InteractivePreviewSeries, ...]

    def __post_init__(self) -> None:
        tab = _required_text(self.tab_id, "tab_id")
        set_id = _required_text(self.set_id, "set_id")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive int.")
        targets = tuple(_required_text(value, "target_set_id") for value in self.target_set_ids)
        if not targets or len(targets) != len(set(targets)) or set_id not in targets:
            raise ValueError("frame target roster is invalid.")
        x_values = _finite_tuple(self.x_values, field="x_values")
        if len(x_values) > 1 and any(current <= previous for previous, current in zip(x_values, x_values[1:])):
            raise ValueError("x_values must be strictly increasing.")
        segment_ids = tuple(int(value) for value in self.segment_ids)
        segment_keys = tuple(_required_text(value, "segment_key") for value in self.segment_keys)
        discontinuity_ids = tuple(int(value) for value in self.discontinuity_ids)
        if not (len(segment_ids) == len(segment_keys) == len(discontinuity_ids) == len(x_values)):
            raise ValueError("Interactive Preview topology must align with x_values.")
        if any(value < 0 for value in segment_ids + discontinuity_ids):
            raise ValueError("Interactive Preview topology IDs must be nonnegative.")
        boundaries = tuple(self.presentation_boundaries)
        if any(not isinstance(value, SimulationPresentationBoundary) for value in boundaries):
            raise TypeError("presentation_boundaries must contain typed boundaries.")
        series = tuple(self.series)
        if not series or any(not isinstance(value, InteractivePreviewSeries) for value in series):
            raise TypeError("series must contain InteractivePreviewSeries values.")
        if any(len(value.values) != len(x_values) for value in series):
            raise ValueError("every series must align with x_values.")
        ids = tuple(value.series_id for value in series)
        if len(ids) != len(set(ids)):
            raise ValueError("Interactive Preview series IDs must be unique.")
        object.__setattr__(self, "tab_id", tab)
        object.__setattr__(self, "set_id", set_id)
        object.__setattr__(self, "target_set_ids", targets)
        object.__setattr__(self, "x_values", x_values)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "presentation_boundaries", boundaries)
        object.__setattr__(self, "series", series)
